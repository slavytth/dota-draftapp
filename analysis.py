# analysis.py
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from config import *
from roles import HERO_DB, HERO_ALIASES
from api import OpenDotaClient, StratzClient
from utils import (
    search_hero, smooth_winrate, candidate_wr_vs_enemy, calculate_advantage,
    sigmoid_scaled, build_form_scores, aggregate_synergy_status, SynergyStatus, synergy_role_weight
)

logger = logging.getLogger(__name__)

@dataclass
class HeroInfo:
    id: int
    name: str
    loc_name: str
    roles: List[int]

@dataclass
class MatchupData:
    enemy_name: str
    games: int
    advantage: float            # сглаженная поправка к среднему WR героя
    raw_wr: float = 0.0         # сырой WR героя в этом матчапе
    base_wr: float = 0.0        # средний WR героя
    warnings: List[str] = field(default_factory=list)

@dataclass
class SynergyData:
    ally_name: str
    games: int
    advantage: float
    raw_wr: float = 0.0
    base_wr: float = 0.0

@dataclass
class CandidateResult:
    hero: HeroInfo
    form_score: float = 0.5
    counter_score: float = 0.5
    synergy_score: float = 0.5
    draft_score: float = 0.5
    final_score: float = 0.5
    form_games: int = 0
    form_eff_games: float = 0.0
    form_wr: float = 0.0
    form_kda: float = 0.0
    base_wr: float = 0.5                       # средний WR героя (OpenDota heroStats)
    expected_wr: Optional[float] = None        # ожидаемый WR против указанных врагов
    matchups: List[MatchupData] = field(default_factory=list)
    synergies: List[SynergyData] = field(default_factory=list)
    synergy_missing: List[str] = field(default_factory=list)   # союзники без данных для ЭТОГО героя
    warnings: List[str] = field(default_factory=list)

@dataclass
class RecommendationResult:
    candidates: List[CandidateResult]
    synergy_status: SynergyStatus
    synergy_detail: str
    matches_used: int
    matches_requested: int
    warnings: List[str] = field(default_factory=list)
    synergy_missing_allies: List[str] = field(default_factory=list)
    counter_source: str = ""      # откуда взяты контрпики: Stratz (высокие ранги) или OpenDota (все ранги)

class DotaAnalyzer:
    def __init__(self, od_client: OpenDotaClient, stratz_client: StratzClient):
        self.od = od_client
        self.stratz = stratz_client

        self.heroes_data = self.od.get_heroes()
        self.check_roles_coverage()

        self.heroes: Dict[int, HeroInfo] = {}
        self.search_map: Dict[str, int] = {}
        for hid, h in self.heroes_data.items():
            db_entry = HERO_DB.get(hid)
            roles = db_entry[2] if db_entry else []
            loc_name = db_entry[1] if db_entry else h['localized_name']
            self.heroes[hid] = HeroInfo(hid, h['name'], loc_name, roles)
            self.search_map[loc_name.lower()] = hid
            self.search_map[h['localized_name'].lower()] = hid
            self.search_map[h['name'].lower().replace('npc_dota_hero_', '')] = hid
        for alias, original in HERO_ALIASES.items():
            if original.lower() in self.search_map:
                self.search_map[alias.lower()] = self.search_map[original.lower()]

        stats = self.od.get_hero_stats()
        self.base_wrs_all: Dict[int, float] = {}
        self.base_wrs_high: Dict[int, float] = {}
        for h in stats:
            picks = sum(h.get(f"{i}_pick", 0) for i in range(1, 9))
            wins = sum(h.get(f"{i}_win", 0) for i in range(1, 9))
            self.base_wrs_all[h['id']] = wins / picks if picks > 0 else 0.5
            # Базовый WR высоких рангов (тот же срез, что и данные Stratz DIVINE_IMMORTAL)
            hp = sum(h.get(f"{i}_pick", 0) for i in HIGH_BRACKETS)
            hw = sum(h.get(f"{i}_win", 0) for i in HIGH_BRACKETS)
            if hp >= MIN_HIGH_BASE_PICKS:
                self.base_wrs_high[h['id']] = hw / hp
        self.base_wrs = self.base_wrs_all
        self.od.cache.flush()   # сохраняем /heroes и /heroStats даже если дальше что-то упадёт

    def check_roles_coverage(self):
        missing = [h['localized_name'] for hid, h in self.heroes_data.items() if hid not in HERO_DB]
        if missing:
            logger.warning(f"Герои без ролей в roles.py ({len(missing)} шт): {', '.join(missing)}")

    def resolve_hero(self, query: str) -> Tuple[Optional[HeroInfo], List[HeroInfo]]:
        return search_hero(query, self.search_map, self.heroes)

    def get_recommendations(self, acc_id: int, role: int, enemies: List[HeroInfo], allies: List[HeroInfo],
                            matches_limit: int, min_games: int, only_played: bool,
                            ally_roles: Optional[Dict[int, int]] = None) -> RecommendationResult:
        global_warnings: List[str] = []
        ally_roles = ally_roles or {}
        conf_e = min(1.0, len(enemies) / 5)     # какая доля вражеского драфта известна
        conf_a = min(1.0, len(allies) / 4)
        if 0 < len(enemies) < MIN_ENEMIES_FOR_FULL_COUNTER:
            global_warnings.append(f"Драфт врагов неполный ({len(enemies)} из 5): совет предварительный, вес контрпика снижен.")
        try:
            matches, matches_used = self.od.get_player_matches(acc_id, matches_limit)
            if matches_used < matches_limit:
                global_warnings.append(
                    f"Найдено подходящих матчей только {matches_used} из {matches_limit} запрошенных.")

            forms = build_form_scores(matches, time.time(), HALF_LIFE_DAYS, PRIOR_MATCHES, SHRINK_K, FORM_GAIN)
            picked_ids = {h.id for h in enemies + allies}

            # --- Контрпики: Stratz (высокие ранги) если доступен для ВСЕХ врагов, иначе OpenDota (все ранги) ---
            enemy_matchups: Dict[int, Dict[int, dict]] = {}
            use_stratz_counters = False
            counter_source = "OpenDota (все ранги)"
            if enemies and self.stratz.token:
                tmp, fail_detail = {}, ""
                for eh in enemies:
                    st, det, vs = self.stratz.get_counters(eh.id)
                    if st != SynergyStatus.OK:
                        fail_detail = f"{eh.loc_name}: {det}"
                        break
                    tmp[eh.id] = {hid2: {'hero_id': hid2, 'games_played': x['matchCount'], 'wins': x['winCount']}
                                  for hid2, x in vs.items()}
                if not fail_detail:
                    enemy_matchups, use_stratz_counters = tmp, True
                    counter_source = f"Stratz ({STRATZ_BRACKETS.replace('_', '+').title()})"
                else:
                    global_warnings.append(f"Контрпики Stratz недоступны ({fail_detail}); использованы данные OpenDota (все ранги).")
            elif enemies:
                global_warnings.append("Нет токена Stratz: контрпики из OpenDota (все ранги, менее точны для высокого рейтинга).")
            if not use_stratz_counters:
                enemy_matchups = {eh.id: {m['hero_id']: m for m in self.od.get_matchups(eh.id)} for eh in enemies}

            # --- Синергия: статус считается ОДИН раз по всем союзникам (aggregate_synergy_status) ---
            ally_synergies: Dict[int, dict] = {}
            per_ally: Dict[str, Tuple[SynergyStatus, str]] = {}
            for ah in allies:
                status, detail, data = self.stratz.get_synergy(ah.id)
                per_ally[ah.loc_name] = (status, detail)
                ally_synergies[ah.id] = data if status == SynergyStatus.OK else {}
            syn_status, syn_detail, missing_allies = aggregate_synergy_status(per_ally)
            if syn_status == SynergyStatus.PARTIAL_DATA:
                global_warnings.append(f"Stratz: данные по союзникам неполные ({syn_detail}).")

            candidates: List[CandidateResult] = []
            for hid, hero in self.heroes.items():
                if hid in picked_ids or not hero.roles or role not in hero.roles:
                    continue
                f = forms.get(hid)
                if only_played and (not f or f['games'] < min_games):
                    continue
                if only_played and f['score'] < POOL_MIN_FORM:
                    continue    # личный пул: только герои с нормальной недавней формой

                c = CandidateResult(hero=hero)
                if f:
                    c.form_score = f['score']
                    c.form_games = f['games']
                    c.form_eff_games = f['eff_games']
                    c.form_wr = f['wr']
                    c.form_kda = f['kda']
                    if c.form_games < 5:
                        c.warnings.append(f"Мало игр на герое ({c.form_games} < 5), оценка формы ненадёжна.")

                base_all = self.base_wrs_all.get(hid, 0.5)
                base_high = self.base_wrs_high.get(hid, base_all)
                # База контрпиков должна быть из того же среза, что и сами матчапы
                base_wr = base_high if use_stratz_counters else base_all
                c.base_wr = base_wr if enemies else base_high
                # Общая сила героя: ожидаемый WR = средний WR + поправка матчапа
                base_term = BASE_STRENGTH_WEIGHT * (base_wr - 0.5)
                syn_term = BASE_STRENGTH_WEIGHT * (base_high - 0.5)     # синергия Stratz тоже по высоким рангам

                if enemies:
                    advs, weights = [], []
                    for eh in enemies:
                        m = enemy_matchups[eh.id].get(hid)
                        if m and m['games_played'] > 0:
                            games = m['games_played']
                            cand_wr = candidate_wr_vs_enemy(m['wins'], games)
                            # Сильное сглаживание к среднему WR героя: малые выборки почти не двигают оценку
                            smooth_cand_wr = smooth_winrate(cand_wr * games, games, MATCHUP_PRIOR_GAMES, base_wr)
                            adv = calculate_advantage(smooth_cand_wr, base_wr)
                            advs.append(adv * games)
                            weights.append(games)
                            md = MatchupData(eh.loc_name, games, adv, raw_wr=cand_wr, base_wr=base_wr)
                            if games < MIN_MATCHUP_GAMES:
                                md.warnings.append(f"Мало матчап-игр ({games})")
                            c.matchups.append(md)
                        else:
                            c.warnings.append(f"Нет данных матчапа против {eh.loc_name}")
                    weighted_adv = sum(advs) / sum(weights) if weights and sum(weights) > 0 else 0.0
                    c.counter_score = sigmoid_scaled(base_term + weighted_adv, COUNTER_SCALE)
                    if weights:
                        c.expected_wr = base_wr + weighted_adv

                if allies:
                    s_advs = []     # (поправка, вес по позиции союзника)
                    for ah in allies:
                        m = ally_synergies.get(ah.id, {}).get(hid)
                        if m and m.get('matchCount', 0) > 0:
                            games = m['matchCount']
                            wr = m['winCount'] / games
                            smooth_wr = smooth_winrate(wr * games, games, SYNERGY_PRIOR_GAMES, base_high)
                            adv = calculate_advantage(smooth_wr, base_high)
                            s_advs.append((adv, synergy_role_weight(role, ally_roles.get(ah.id))))
                            c.synergies.append(SynergyData(ah.loc_name, games, adv, raw_wr=wr, base_wr=base_high))
                    if s_advs:
                        c.synergy_score = sigmoid_scaled(syn_term + sum(a * w for a, w in s_advs) / sum(w for _, w in s_advs), SYNERGY_SCALE)
                    have = {s.ally_name for s in c.synergies}
                    c.synergy_missing = [ah.loc_name for ah in allies if ah.loc_name not in have]

                # Драфт: если синергии нет, опираемся только на контрпик (и наоборот)
                if enemies and c.synergies:
                    wc, ws = DRAFT_COUNTER_RATIO * conf_e, DRAFT_SYNERGY_RATIO * conf_a
                    c.draft_score = (wc * c.counter_score + ws * c.synergy_score) / (wc + ws)
                elif enemies:
                    c.draft_score = c.counter_score
                elif c.synergies:
                    c.draft_score = c.synergy_score

                c.final_score = c.form_score * WEIGHT_FORM + c.draft_score * WEIGHT_DRAFT
                candidates.append(c)

            candidates.sort(key=lambda x: x.final_score, reverse=True)
            return RecommendationResult(candidates, syn_status, syn_detail, matches_used,
                                        matches_limit, global_warnings, missing_allies, counter_source)
        finally:
            self.od.cache.flush()
            sc = getattr(self.stratz, 'cache', None)
            if sc is not None:
                sc.flush()


def select_recommendations(candidates: List[CandidateResult], n_pool: int = 2, n_total: int = 3,
                           min_games: int = 3) -> List[Tuple[str, CandidateResult]]:
    """
    Итоговая выдача: до n_pool героев из личного пула (есть игры и форма не ниже POOL_MIN_FORM; ранг по итоговому
    баллу 40/60) + остальные места занимают «мета-пики»: лучшие по драфту (контрпик + синергия + сила героя),
    независимо от практики. Если в пуле меньше n_pool героев, свободные места достаются мета-пикам.
    """
    pool = [c for c in candidates if c.form_games >= min_games and c.form_score >= POOL_MIN_FORM]
    pool.sort(key=lambda c: c.final_score, reverse=True)
    picks = [("pool", c) for c in pool[:n_pool]]
    chosen = {c.hero.id for _, c in picks}
    rest = sorted((c for c in candidates if c.hero.id not in chosen),
                  key=lambda c: (c.draft_score, c.base_wr), reverse=True)
    picks += [("meta", c) for c in rest[:max(0, n_total - len(picks))]]
    return picks
