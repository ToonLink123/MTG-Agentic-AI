//! Ring's goldfish engine: solitaire Commander games against three idle opponents.
//!
//! Reads a compiled deck (mtg_agentic.goldfish.compile_deck) as JSON on stdin and writes
//! counters JSON on stdout, which mtg_agentic.goldfish.summarize turns into a report. Every
//! game is seeded from (seed, game number) with splitmix64, so the counters for a seed are the
//! same on any number of threads. `python -m mtg_agentic setup` builds it.
//!
//! Usage: goldfish-rs --games N --turns T --seed S [--threads K] < compiled.json

use serde::{Deserialize, Serialize};
use std::io::Read;

const OPPONENTS: usize = 3;
const STARTING_LIFE: i64 = 40;
const COMMANDER_DAMAGE: i64 = 21;
const TAPPED: u8 = 1;
const FAST: u8 = 2;
const SLOW: u8 = 3;
const CHECK: u8 = 4;
const UNTAPPED: u8 = 0;

#[derive(Deserialize)]
struct Piece {
    mv: i64,
    pips: Vec<u8>,
    generic: usize,
    total: usize,
    castable: bool,
    land: bool,
    back_land: bool,
    colors: u8,
    land_types: u8,
    enter: u8,
    check_types: u8,
    fetch: Vec<usize>,
    fetch_tapped: bool,
    creature: bool,
    power: i64,
    haste: bool,
    mana: Vec<u8>,
    ramp: Vec<usize>,
    ramp_battlefield: usize,
    ramp_hand: usize,
    ramp_tapped: bool,
    treasures: usize,
    draw: usize,
    upkeep_draw: usize,
    burn_each: i64,
    burn_one: i64,
    token_power: i64,
    token_haste: bool,
    #[serde(rename = "class")]
    cls: i64,
}

#[derive(Deserialize)]
struct Deck {
    commander: usize,
    library: Vec<usize>,
    pieces: Vec<Piece>,
}

#[derive(Serialize, Clone)]
struct Counters {
    games: u64,
    turns: usize,
    on_play: u64,
    mulligans: Vec<u64>,
    opening_lands: Vec<u64>,
    land_drops: Vec<u64>,
    lands_on_curve: Vec<u64>,
    mana_available: Vec<u64>,
    mana_spent: Vec<u64>,
    color_screw: Vec<u64>,
    commander_turn: Vec<u64>,
    first_kill: Vec<u64>,
    kill: Vec<u64>,
    drawn: Vec<u64>,
    cast: Vec<u64>,
    cast_turn_sum: Vec<u64>,
    stranded: Vec<u64>,
}

impl Counters {
    fn new(pieces: usize, turns: usize) -> Self {
        Counters {
            games: 0,
            turns,
            on_play: 0,
            mulligans: vec![0; 8],
            opening_lands: vec![0; 8],
            land_drops: vec![0; turns],
            lands_on_curve: vec![0; turns],
            mana_available: vec![0; turns],
            mana_spent: vec![0; turns],
            color_screw: vec![0; turns],
            commander_turn: vec![0; turns + 1],
            first_kill: vec![0; turns + 1],
            kill: vec![0; turns + 1],
            drawn: vec![0; pieces],
            cast: vec![0; pieces],
            cast_turn_sum: vec![0; pieces],
            stranded: vec![0; pieces],
        }
    }

    fn merge(&mut self, other: &Counters) {
        fn add(a: &mut [u64], b: &[u64]) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        self.games += other.games;
        self.on_play += other.on_play;
        add(&mut self.mulligans, &other.mulligans);
        add(&mut self.opening_lands, &other.opening_lands);
        add(&mut self.land_drops, &other.land_drops);
        add(&mut self.lands_on_curve, &other.lands_on_curve);
        add(&mut self.mana_available, &other.mana_available);
        add(&mut self.mana_spent, &other.mana_spent);
        add(&mut self.color_screw, &other.color_screw);
        add(&mut self.commander_turn, &other.commander_turn);
        add(&mut self.first_kill, &other.first_kill);
        add(&mut self.kill, &other.kill);
        add(&mut self.drawn, &other.drawn);
        add(&mut self.cast, &other.cast);
        add(&mut self.cast_turn_sum, &other.cast_turn_sum);
        add(&mut self.stranded, &other.stranded);
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn game_seed(seed: u64, game: u64) -> u64 {
    seed.wrapping_mul(1_000_003).wrapping_add(game)
}

fn shuffle(items: &mut [usize], rng: &mut SplitMix64) {
    for i in (1..items.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        items.swap(i, j);
    }
}

fn pop(mask: u8) -> u32 {
    mask.count_ones()
}

/// Pay colored pips with the least flexible mana that fits, then generic the same way;
/// Treasure last. The new pool and Treasure count, or None if the cost can't be paid.
fn pay(pool: &[u8], treasures: usize, pips: &[u8], generic: usize) -> Option<(Vec<u8>, usize)> {
    let mut pool = pool.to_vec();
    let mut treasures = treasures;
    for &pip in pips {
        let mut best: Option<usize> = None;
        let mut best_pop = 99;
        for (i, &unit) in pool.iter().enumerate() {
            if unit & pip != 0 && pop(unit) < best_pop {
                best = Some(i);
                best_pop = pop(unit);
            }
        }
        match best {
            Some(i) => {
                pool.remove(i);
            }
            None if treasures > 0 => treasures -= 1,
            None => return None,
        }
    }
    if pool.len() + treasures < generic {
        return None;
    }
    for _ in 0..generic {
        if !pool.is_empty() {
            let mut best = 0;
            let mut best_pop = 99;
            for (i, &unit) in pool.iter().enumerate() {
                if pop(unit) < best_pop {
                    best = i;
                    best_pop = pop(unit);
                }
            }
            pool.remove(best);
        } else {
            treasures -= 1;
        }
    }
    Some((pool, treasures))
}

struct Creature {
    power: i64,
    entered: usize,
    haste: bool,
    commander: bool,
}

struct Game<'a> {
    p: &'a [Piece],
    c: &'a mut Counters,
    rng: SplitMix64,
    turns: usize,
    commander: usize,
    library: &'a [usize],
    lib: Vec<usize>,
    hand: Vec<usize>,
    lands: Vec<usize>,
    rocks: Vec<usize>,
    dorks: Vec<(usize, usize)>,
    creatures: Vec<Creature>,
    pool: Vec<u8>,
    treasures: usize,
    upkeep: usize,
    life: [i64; OPPONENTS],
    cmd: [i64; OPPONENTS],
    commander_turn: usize,
    t: usize,
}

type Choice = (usize, usize, Vec<u8>, usize);

impl<'a> Game<'a> {
    fn play(&mut self) {
        let on_play = self.rng.below(4) == 0;
        self.c.on_play += on_play as u64;
        let mulligans = self.mulligan();
        self.c.mulligans[mulligans.min(7)] += 1;
        let lands = self.hand.iter().filter(|&&i| self.p[i].land || self.p[i].back_land).count();
        self.c.opening_lands[lands.min(7)] += 1;
        for k in 0..self.hand.len() {
            self.c.drawn[self.hand[k]] += 1;
        }
        let (mut first_kill, mut kill) = (0, 0);
        for t in 1..=self.turns {
            self.t = t;
            self.draw(self.upkeep);
            if t > 1 || !on_play {
                self.draw(1);
            }
            let mut pool = self.base_pool();
            let pos = self.choose_land(&pool);
            if let Some(pos) = pos {
                self.c.land_drops[t - 1] += 1;
                let unit = self.play_land(pos);
                if unit != 0 {
                    pool.push(unit);
                }
            }
            self.pool = pool;
            self.c.mana_available[t - 1] += (self.pool.len() + self.treasures) as u64;
            let spent = self.cast_all();
            self.c.mana_spent[t - 1] += spent as u64;
            self.c.color_screw[t - 1] += self.color_screwed() as u64;
            self.combat();
            self.c.lands_on_curve[t - 1] += (self.lands.len() >= t) as u64;
            let dead = (0..OPPONENTS)
                .filter(|&k| self.life[k] <= 0 || self.cmd[k] >= COMMANDER_DAMAGE)
                .count();
            if dead > 0 && first_kill == 0 {
                first_kill = t;
            }
            if dead == OPPONENTS && kill == 0 {
                kill = t;
            }
        }
        let never = self.turns + 1;
        let turn = if self.commander_turn > 0 { self.commander_turn } else { never };
        self.c.commander_turn[turn - 1] += 1;
        self.c.first_kill[(if first_kill > 0 { first_kill } else { never }) - 1] += 1;
        self.c.kill[(if kill > 0 { kill } else { never }) - 1] += 1;
        for k in 0..self.hand.len() {
            let i = self.hand[k];
            if !self.p[i].land {
                self.c.stranded[i] += 1;
            }
        }
        self.c.games += 1;
    }

    fn is_land_card(&self, i: usize) -> bool {
        self.p[i].land || self.p[i].back_land
    }

    fn mulligan(&mut self) -> usize {
        let mut mulligans: usize = 0;
        let bottom;
        loop {
            self.lib.clear();
            self.lib.extend_from_slice(self.library);
            shuffle(&mut self.lib, &mut self.rng);
            self.hand.clear();
            for _ in 0..7 {
                let card = self.lib.pop().expect("at least seven cards");
                self.hand.push(card);
            }
            let b = mulligans.saturating_sub(1);
            let lands = self.hand.iter().filter(|&&i| self.is_land_card(i)).count();
            if (2..=5).contains(&lands) || b >= 2 {
                bottom = b;
                break;
            }
            mulligans += 1;
        }
        for _ in 0..bottom {
            self.bottom_one();
        }
        mulligans
    }

    fn bottom_one(&mut self) {
        let lands: Vec<usize> =
            (0..self.hand.len()).filter(|&k| self.is_land_card(self.hand[k])).collect();
        let k = if lands.len() > 3 {
            let mut best = lands[0];
            let mut best_key = (false, i64::MIN, 0usize);
            for &k in &lands {
                let p = &self.p[self.hand[k]];
                let key = (p.enter == TAPPED, -(pop(p.colors) as i64), k);
                if key > best_key {
                    best = k;
                    best_key = key;
                }
            }
            best
        } else {
            let spells: Vec<usize> = (0..self.hand.len()).filter(|k| !lands.contains(k)).collect();
            if spells.is_empty() {
                *lands.last().expect("a card to bottom")
            } else {
                let mut best = spells[0];
                let mut best_key = (i64::MIN, 0usize);
                for &k in &spells {
                    let key = (self.p[self.hand[k]].mv, k);
                    if key > best_key {
                        best = k;
                        best_key = key;
                    }
                }
                best
            }
        };
        let card = self.hand.remove(k);
        self.lib.insert(0, card);
    }

    fn draw(&mut self, n: usize) {
        for _ in 0..n {
            match self.lib.pop() {
                Some(i) => {
                    self.hand.push(i);
                    self.c.drawn[i] += 1;
                }
                None => return,
            }
        }
    }

    fn base_pool(&self) -> Vec<u8> {
        let mut pool: Vec<u8> =
            self.lands.iter().map(|&i| self.p[i].colors).filter(|&c| c != 0).collect();
        for &i in &self.rocks {
            pool.extend_from_slice(&self.p[i].mana);
        }
        for &(i, entered) in &self.dorks {
            if entered < self.t {
                pool.extend_from_slice(&self.p[i].mana);
            }
        }
        pool
    }

    fn enters_untapped(&self, i: usize) -> bool {
        let p = &self.p[i];
        match p.enter {
            UNTAPPED => true,
            FAST => self.lands.len() <= 2,
            SLOW => self.lands.len() >= 2,
            CHECK => self.lands.iter().any(|&l| self.p[l].land_types & p.check_types != 0),
            _ => false,
        }
    }

    fn colors_in_play(&self) -> u8 {
        self.lands.iter().fold(0, |acc, &i| acc | self.p[i].colors)
    }

    /// The library land that adds the most new colors, then one that enters untapped.
    fn fetch_target(&self, targets: &[usize]) -> Option<usize> {
        let in_play = self.colors_in_play();
        let mut best = None;
        let mut best_key = (-1i64, -1i64);
        for &target in targets {
            if !self.lib.contains(&target) {
                continue;
            }
            let key = (
                pop(self.p[target].colors & !in_play) as i64,
                self.enters_untapped(target) as i64,
            );
            if key > best_key {
                best = Some(target);
                best_key = key;
            }
        }
        best
    }

    fn take_from_library(&mut self, i: usize) {
        if let Some(k) = self.lib.iter().rposition(|&x| x == i) {
            self.lib.remove(k);
        }
    }

    fn land_outcome(&self, i: usize) -> (u8, bool) {
        let p = &self.p[i];
        if !p.fetch.is_empty() {
            return match self.fetch_target(&p.fetch) {
                None => (0, false),
                Some(target) => {
                    (self.p[target].colors, !p.fetch_tapped && self.enters_untapped(target))
                }
            };
        }
        (p.colors, self.enters_untapped(i))
    }

    fn choose_land(&self, pool: &[u8]) -> Option<usize> {
        let mut candidates: Vec<usize> =
            (0..self.hand.len()).filter(|&k| self.p[self.hand[k]].land).collect();
        if candidates.is_empty() {
            candidates = (0..self.hand.len()).filter(|&k| self.p[self.hand[k]].back_land).collect();
        }
        let mut distinct: Vec<usize> = Vec::new();
        let mut seen: Vec<usize> = Vec::new();
        for k in candidates {
            if !seen.contains(&self.hand[k]) {
                seen.push(self.hand[k]);
                distinct.push(k);
            }
        }
        if distinct.len() <= 1 {
            return distinct.first().copied();
        }
        let in_play = self.colors_in_play();
        let mut best = None;
        let mut best_key = (-1i64, -1i64, -1i64);
        for &k in &distinct {
            let (color, untapped) = self.land_outcome(self.hand[k]);
            let mut trial = pool.to_vec();
            if untapped && color != 0 {
                trial.push(color);
            }
            let mut rest = self.hand.clone();
            rest.remove(k);
            let spend = self.plan(rest, trial) as i64;
            // Most mana spent; on a tie, play the tapped land now; then the most new colors.
            let key = (spend, if untapped { 0 } else { 1 }, pop(color & !in_play) as i64);
            if key > best_key {
                best = Some(k);
                best_key = key;
            }
        }
        best
    }

    /// Play a land from hand; the mana it adds this turn, or 0.
    fn play_land(&mut self, pos: usize) -> u8 {
        let i = self.hand.remove(pos);
        self.c.cast[i] += 1;
        self.c.cast_turn_sum[i] += self.t as u64;
        let pieces = self.p;
        let p = &pieces[i];
        if !p.fetch.is_empty() {
            let Some(target) = self.fetch_target(&p.fetch) else {
                return 0;
            };
            let untapped = !p.fetch_tapped && self.enters_untapped(target);
            self.take_from_library(target);
            self.lands.push(target);
            shuffle(&mut self.lib, &mut self.rng);
            return if untapped { self.p[target].colors } else { 0 };
        }
        let untapped = self.enters_untapped(i);
        self.lands.push(i);
        if untapped {
            p.colors
        } else {
            0
        }
    }

    /// The first affordable spell by class, then mana value, then hand order.
    fn pick(&self, hand: &[usize], pool: &[u8], treasures: usize, commander: Option<usize>) -> Option<Choice> {
        let budget = pool.len() + treasures;
        let mut options: Vec<(i64, i64, usize)> = Vec::new();
        for (k, &i) in hand.iter().enumerate() {
            let p = &self.p[i];
            if p.castable && p.total <= budget {
                options.push((p.cls, -p.mv, k));
            }
        }
        if let Some(c) = commander {
            if self.p[c].total <= budget {
                options.push((self.p[c].cls, -self.p[c].mv, hand.len()));
            }
        }
        options.sort();
        for &(_, _, k) in &options {
            let i = if k == hand.len() { commander.expect("commander option") } else { hand[k] };
            if let Some((pool, treasures)) = pay(pool, treasures, &self.p[i].pips, self.p[i].generic) {
                return Some((k, i, pool, treasures));
            }
        }
        None
    }

    fn open_commander(&self) -> Option<usize> {
        if self.commander_turn > 0 {
            None
        } else {
            Some(self.commander)
        }
    }

    /// Mana a turn would spend: the cast loop without draws, searches or Treasure.
    fn plan(&self, mut hand: Vec<usize>, mut pool: Vec<u8>) -> usize {
        let mut treasures = self.treasures;
        let mut commander = self.open_commander();
        let mut spent = 0;
        while let Some((k, i, new_pool, new_treasures)) = self.pick(&hand, &pool, treasures, commander) {
            pool = new_pool;
            treasures = new_treasures;
            let p = &self.p[i];
            spent += p.total;
            if k == hand.len() {
                commander = None;
            } else {
                hand.remove(k);
            }
            if !p.mana.is_empty() && !p.creature {
                pool.extend_from_slice(&p.mana);
            }
        }
        spent
    }

    /// After casting, the mana left over could pay for a spell in hand by count but not by
    /// its colors.
    fn color_screwed(&self) -> bool {
        let budget = self.pool.len() + self.treasures;
        let mut options: Vec<usize> =
            self.hand.iter().copied().filter(|&i| self.p[i].castable).collect();
        if let Some(c) = self.open_commander() {
            options.push(c);
        }
        options.iter().any(|&i| {
            self.p[i].total <= budget
                && pay(&self.pool, self.treasures, &self.p[i].pips, self.p[i].generic).is_none()
        })
    }

    fn cast_all(&mut self) -> usize {
        let mut spent = 0;
        while let Some((k, i, pool, treasures)) =
            self.pick(&self.hand, &self.pool, self.treasures, self.open_commander())
        {
            self.pool = pool;
            self.treasures = treasures;
            let commander = k == self.hand.len();
            if commander {
                self.commander_turn = self.t;
            } else {
                self.hand.remove(k);
            }
            spent += self.p[i].total;
            self.c.cast[i] += 1;
            self.c.cast_turn_sum[i] += self.t as u64;
            self.resolve(i, commander);
        }
        spent
    }

    fn resolve(&mut self, i: usize, commander: bool) {
        let pieces = self.p;
        let p = &pieces[i];
        let t = self.t;
        if !p.mana.is_empty() {
            if p.creature {
                self.dorks.push((i, t));
            } else {
                self.rocks.push(i);
                self.pool.extend_from_slice(&p.mana);
            }
        }
        if p.creature {
            self.creatures.push(Creature { power: p.power, entered: t, haste: p.haste, commander });
        }
        let mut searched = false;
        for _ in 0..p.ramp_battlefield {
            let Some(target) = self.fetch_target(&p.ramp) else { break };
            let untapped = !p.ramp_tapped && self.enters_untapped(target);
            self.take_from_library(target);
            self.lands.push(target);
            searched = true;
            if untapped && self.p[target].colors != 0 {
                self.pool.push(self.p[target].colors);
            }
        }
        for _ in 0..p.ramp_hand {
            let Some(target) = self.fetch_target(&p.ramp) else { break };
            self.take_from_library(target);
            self.hand.push(target);
            self.c.drawn[target] += 1;
            searched = true;
        }
        if searched {
            shuffle(&mut self.lib, &mut self.rng);
        }
        self.treasures += p.treasures;
        self.draw(p.draw);
        if p.burn_each != 0 {
            for k in 0..OPPONENTS {
                if self.alive(k) {
                    self.life[k] -= p.burn_each;
                }
            }
        }
        if p.burn_one != 0 {
            if let Some(target) = self.weakest() {
                self.life[target] -= p.burn_one;
            }
        }
        if p.token_power != 0 {
            self.creatures.push(Creature {
                power: p.token_power,
                entered: t,
                haste: p.token_haste,
                commander: false,
            });
        }
        self.upkeep += p.upkeep_draw;
    }

    fn alive(&self, k: usize) -> bool {
        self.life[k] > 0 && self.cmd[k] < COMMANDER_DAMAGE
    }

    fn weakest(&self) -> Option<usize> {
        let mut target: Option<usize> = None;
        for k in 0..OPPONENTS {
            if self.alive(k) && target.map_or(true, |t| self.life[k] < self.life[t]) {
                target = Some(k);
            }
        }
        target
    }

    fn combat(&mut self) {
        let mut attackers: Vec<(i64, bool)> = self
            .creatures
            .iter()
            .filter(|c| c.power > 0 && (c.entered < self.t || c.haste))
            .map(|c| (c.power, c.commander))
            .collect();
        attackers.sort_by_key(|&(power, _)| std::cmp::Reverse(power));
        for (power, commander) in attackers {
            let Some(target) = self.weakest() else { return };
            self.life[target] -= power;
            if commander {
                self.cmd[target] += power;
            }
        }
    }
}

fn play_games(deck: &Deck, first: u64, count: u64, turns: usize, seed: u64) -> Counters {
    let mut counters = Counters::new(deck.pieces.len(), turns);
    for game in first..first + count {
        let mut g = Game {
            p: &deck.pieces,
            c: &mut counters,
            rng: SplitMix64(game_seed(seed, game)),
            turns,
            commander: deck.commander,
            library: &deck.library,
            lib: Vec::with_capacity(deck.library.len()),
            hand: Vec::with_capacity(16),
            lands: Vec::with_capacity(16),
            rocks: Vec::new(),
            dorks: Vec::new(),
            creatures: Vec::new(),
            pool: Vec::new(),
            treasures: 0,
            upkeep: 0,
            life: [STARTING_LIFE; OPPONENTS],
            cmd: [0; OPPONENTS],
            commander_turn: 0,
            t: 0,
        };
        g.play();
    }
    counters
}

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|k| args.get(k + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u64 = arg(&args, "--games", "10000").parse().expect("--games");
    let turns: usize = arg(&args, "--turns", "10").parse().expect("--turns");
    let seed = arg(&args, "--seed", "1").parse::<i64>().expect("--seed") as u64;
    let mut threads: usize = arg(&args, "--threads", "1").parse().expect("--threads");
    if threads == 0 {
        threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    }
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("read stdin");
    let deck: Deck = serde_json::from_str(&input).expect("compiled deck JSON");
    if deck.library.len() < 7 || turns == 0 {
        eprintln!("A goldfish needs at least seven cards and one turn");
        std::process::exit(2);
    }
    let threads = threads.clamp(1, games.max(1) as usize);
    let chunk = games / threads as u64;
    let extra = games % threads as u64;
    let mut total = Counters::new(deck.pieces.len(), turns);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut first = 0;
        for k in 0..threads as u64 {
            let count = chunk + u64::from(k < extra);
            let deck = &deck;
            handles.push(scope.spawn(move || play_games(deck, first, count, turns, seed)));
            first += count;
        }
        for handle in handles {
            total.merge(&handle.join().expect("worker thread"));
        }
    });
    println!("{}", serde_json::to_string(&total).expect("serialize counters"));
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u8 = 1;
    const U: u8 = 2;
    const G: u8 = 16;

    #[test]
    fn generator_matches_reference_splitmix64() {
        // First outputs of splitmix64 seeded with 0, as published with the algorithm.
        let mut rng = SplitMix64(0);
        assert_eq!(rng.next(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn shuffle_is_a_permutation_fixed_by_the_seed() {
        let mut a: Vec<usize> = (0..99).collect();
        let mut b = a.clone();
        shuffle(&mut a, &mut SplitMix64(game_seed(7, 3)));
        shuffle(&mut b, &mut SplitMix64(game_seed(7, 3)));
        assert_eq!(a, b);
        assert_ne!(a, (0..99).collect::<Vec<_>>());
        a.sort();
        assert_eq!(a, (0..99).collect::<Vec<_>>());
    }

    #[test]
    fn payment_uses_least_flexible_mana_then_treasure() {
        assert_eq!(pay(&[G | U, G], 0, &[G], 1), Some((vec![], 0)));
        assert_eq!(pay(&[G | U, G], 0, &[G], 0), Some((vec![G | U], 0)));
        assert_eq!(pay(&[G | U, U], 0, &[U, G], 0), Some((vec![], 0)));
        assert_eq!(pay(&[G, G], 0, &[U], 0), None);
        assert_eq!(pay(&[G, G], 1, &[U], 1), Some((vec![G], 0)));
        assert_eq!(pay(&[G], 0, &[], 2), None);
        assert_eq!(pay(&[W | U, W], 0, &[W, W | U], 0), Some((vec![], 0)));
    }
}
