# tests.py — offline-тесты (без сети). Запуск: python -m unittest tests -v
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import requests

import analysis
import cli
from analysis import DotaAnalyzer, select_recommendations
from api import (OpenDotaClient, StratzClient, AtomicCache,
                 ApiUnavailableError, PrivateProfileError, EmptyHistoryError)
from config import COUNTER_SCALE, SYNERGY_SCALE, MATCHUP_PRIOR_GAMES
from utils import (
    convert_id64_to_id32, is_win, calc_kda, smooth_winrate, candidate_wr_vs_enemy,
    calculate_advantage, parse_synergy_response, SynergyStatus, calculate_absolute_form,
    build_form_scores, search_hero, contribution_report, aggregate_synergy_status,
    sigmoid_scaled, calculate_decay_weight, synergy_role_weight,
)

NOW = 86400 * 100


# ============================ ЧИСТАЯ МАТЕМАТИКА ============================
class TestMath(unittest.TestCase):
    def test_id_conversion(self):
        self.assertEqual(convert_id64_to_id32(76561198038289410), 78023682)
        self.assertEqual(convert_id64_to_id32(78023682), 78023682)

    def test_is_win_all_slots(self):
        self.assertTrue(is_win(0, True));    self.assertFalse(is_win(0, False))
        self.assertTrue(is_win(127, True));  self.assertFalse(is_win(127, False))
        self.assertTrue(is_win(128, False)); self.assertFalse(is_win(128, True))
        self.assertTrue(is_win(255, False)); self.assertFalse(is_win(255, True))

    def test_kda(self):
        self.assertEqual(calc_kda(10, 5, 10), 4.0)
        self.assertEqual(calc_kda(10, 0, 10), 20.0)

    def test_smooth_winrate(self):
        self.assertEqual(smooth_winrate(0, 0, 5, 0.5), 0.5)
        self.assertAlmostEqual(smooth_winrate(3, 3, 5, 0.5), 0.6875)
        self.assertAlmostEqual(smooth_winrate(50, 100, 5, 0.5), 52.5 / 105)

    def test_candidate_wr_and_advantage(self):
        self.assertAlmostEqual(candidate_wr_vs_enemy(60, 100), 0.40)
        self.assertAlmostEqual(calculate_advantage(0.60, 0.55), 0.05)

    def test_decay(self):
        self.assertAlmostEqual(calculate_decay_weight(0, 0, 30), 1.0)
        self.assertAlmostEqual(calculate_decay_weight(0, 86400 * 30, 30), 0.5)
        self.assertAlmostEqual(calculate_decay_weight(0, 86400 * 60, 30), 0.25)

    def test_sigmoid_scale_points(self):
        self.assertAlmostEqual(sigmoid_scaled(0.0, 0.03), 0.5)
        self.assertAlmostEqual(sigmoid_scaled(0.03, 0.03), 0.7311, places=3)
        self.assertAlmostEqual(sigmoid_scaled(0.02, 0.03), 0.6608, places=3)

    def test_absolute_form(self):
        self.assertAlmostEqual(calculate_absolute_form(1.0, 20.0, 0.1, 1, 10, 1.5), 0.5518, places=3)
        self.assertAlmostEqual(calculate_absolute_form(1.0, 5.0, 0.3, 3, 10, 1.5), 0.638, places=3)
        self.assertAlmostEqual(calculate_absolute_form(0.3, 1.5, 0.2, 10, 10, 1.5), 0.320, places=3)


# ============================ ФОРМА (build_form_scores) ============================
def mk(hero, days_ago, win=True, k=10, d=2, a=10):
    return {'hero_id': hero, 'start_time': NOW - 86400 * days_ago, 'player_slot': 0,
            'radiant_win': win, 'kills': k, 'deaths': d, 'assists': a}

class TestFormScores(unittest.TestCase):
    def test_freshness_win_with_good_kda(self):
        forms = build_form_scores([mk(1, 80), mk(2, 1)], NOW, 30.0, 5, 10, 1.5)
        old, new = forms[1], forms[2]
        self.assertLess(old['eff_games'], new['eff_games'])
        self.assertGreater(old['score'], 0.5)            # победа с хорошим KDA -> выше нейтрали
        self.assertGreater(new['score'], 0.5)
        self.assertLess(old['score'], new['score'])      # старая игра ближе к 0.5, чем вчерашняя

    def test_loss_below_neutral(self):
        forms = build_form_scores([mk(1, 1, win=False, k=2, d=8, a=3)], NOW, 30.0, 5, 10, 1.5)
        self.assertLess(forms[1]['score'], 0.5)

    def test_empty(self):
        self.assertEqual(build_form_scores([], NOW, 30.0, 5, 10, 1.5), {})

    def test_regression_one_hero_not_exploding(self):
        # 29 игр на нейтральном герое и 1 на втором: форма не должна улетать в 0.99 / 0.01
        matches = [mk(1, 1, win=(i % 2 == 0), k=5, d=5, a=10) for i in range(29)] + [mk(2, 1)]
        forms = build_form_scores(matches, NOW, 30.0, 5, 10, 1.5)
        for f in forms.values():
            self.assertTrue(0.3 < f['score'] < 0.7)

    def test_unknown_hero_zero_skipped(self):
        forms = build_form_scores([mk(0, 1), mk(5, 1)], NOW, 30.0, 5, 10, 1.5)
        self.assertEqual(list(forms), [5])


# ============================ ИМЕНА ============================
class TestSearch(unittest.TestCase):
    def setUp(self):
        self.m = {"phantom assassin": 44, "pa": 44, "фантомка": 44, "rubick": 86, "pudge": 14}
        self.h = {44: "PA", 86: "Rubick", 14: "Pudge"}

    def test_exact_alias_abbrev(self):
        self.assertEqual(search_hero("Rubick", self.m, self.h)[0], "Rubick")
        self.assertEqual(search_hero("Фантомка", self.m, self.h)[0], "PA")
        self.assertEqual(search_hero("pa", self.m, self.h)[0], "PA")

    def test_typo_suggests(self):
        exact, suggs = search_hero("pudg", self.m, self.h)
        self.assertIsNone(exact)
        self.assertIn("Pudge", suggs)

    def test_short_no_fuzzy(self):
        exact, suggs = search_hero("po", self.m, self.h)
        self.assertIsNone(exact)
        self.assertEqual(suggs, [])


# ============================ STRATZ: парсинг и статусы ============================
class TestSynergyParsing(unittest.TestCase):
    def test_parse(self):
        ok_d = {"data": {"heroStats": {"matchUp": {"with": [{"heroId2": 2}]}}}}
        ok_l = {"data": {"heroStats": {"matchUp": [{"with": [{"heroId2": 2}]}]}}}
        self.assertEqual(parse_synergy_response(ok_d)[0], SynergyStatus.OK)
        self.assertEqual(parse_synergy_response(ok_l)[0], SynergyStatus.OK)
        self.assertEqual(parse_synergy_response({"data": {"heroStats": {"matchUp": {"with": []}}}})[0], SynergyStatus.NO_DATA)
        self.assertEqual(parse_synergy_response({"data": {"heroStats": {"matchUp": {}}}})[0], SynergyStatus.NO_DATA)
        self.assertEqual(parse_synergy_response({"data": {"heroStats": {"matchUp": []}}})[0], SynergyStatus.NO_DATA)
        self.assertEqual(parse_synergy_response({"errors": [{"message": "x"}], "data": None})[0], SynergyStatus.FAILED)
        self.assertEqual(parse_synergy_response({"data": None})[0], SynergyStatus.FAILED)  # data:null без errors
        self.assertEqual(parse_synergy_response({"data": {"heroStats": {"matchUp": {"with": None}}}})[0], SynergyStatus.NO_DATA)

    def test_parse_real_stratz_sample(self):
        # Реальный ответ Stratz Explorer (heroId=1): matchUp приходит СПИСКОМ из одного элемента с with[...]
        sample = {"data": {"heroStats": {"matchUp": [{"with": [
            {"heroId2": 145, "matchCount": 1860, "winCount": 921},
            {"heroId2": 56, "matchCount": 2560, "winCount": 1393},
            {"heroId2": 112, "matchCount": 6864, "winCount": 3629}]}]}}}
        st, detail, data = parse_synergy_response(sample)
        self.assertEqual(st, SynergyStatus.OK)
        self.assertEqual(sorted(data), [56, 112, 145])
        self.assertEqual(data[112]["matchCount"], 6864)
        self.assertAlmostEqual(data[112]["winCount"] / data[112]["matchCount"], 0.5287, places=3)

    def test_parse_matchup_with_and_vs(self):
        # Реальная структура из Explorer (heroId=1, DIVINE_IMMORTAL): Anti-Mage vs Medusa (94): 192 из 288
        sample = {"data": {"heroStats": {"matchUp": [{
            "with": [{"heroId2": 40, "matchCount": 468, "winCount": 252}],
            "vs": [{"heroId2": 94, "matchCount": 288, "winCount": 192}, {"heroId2": 42, "matchCount": 1301, "winCount": 687}]}]}}}
        from utils import parse_matchup_response
        st, _, w, v = parse_matchup_response(sample)
        self.assertEqual(st, SynergyStatus.OK)
        self.assertEqual(sorted(w), [40]); self.assertEqual(sorted(v), [42, 94])
        self.assertAlmostEqual(v[94]["winCount"] / v[94]["matchCount"], 0.6667, places=3)

    def test_parse_matchup_error_message_in_detail(self):
        from utils import parse_matchup_response
        st, detail, _, _ = parse_matchup_response({"errors": [{"message": "Unknown argument take"}], "data": None})
        self.assertEqual(st, SynergyStatus.FAILED)
        self.assertIn("Unknown argument take", detail)

    def test_aggregate_all_ok(self):
        st, _, miss = aggregate_synergy_status({"A": (SynergyStatus.OK, ""), "B": (SynergyStatus.OK, "")})
        self.assertEqual((st, miss), (SynergyStatus.OK, []))

    def test_aggregate_empty_allies(self):
        self.assertEqual(aggregate_synergy_status({})[0], SynergyStatus.OK)

    def test_aggregate_ok_plus_failed_is_partial(self):
        st, detail, miss = aggregate_synergy_status({"A": (SynergyStatus.OK, ""), "B": (SynergyStatus.FAILED, "timeout")})
        self.assertEqual(st, SynergyStatus.PARTIAL_DATA)
        self.assertEqual(miss, ["B"])
        self.assertIn("timeout", detail)

    def test_aggregate_ok_plus_nodata_is_partial(self):
        st, _, miss = aggregate_synergy_status({"A": (SynergyStatus.OK, ""), "B": (SynergyStatus.NO_DATA, "x")})
        self.assertEqual((st, miss), (SynergyStatus.PARTIAL_DATA, ["B"]))

    def test_aggregate_all_failed_is_failed(self):   # раньше давало PARTIAL
        st, detail, miss = aggregate_synergy_status({"A": (SynergyStatus.FAILED, "e1"), "B": (SynergyStatus.FAILED, "e2")})
        self.assertEqual(st, SynergyStatus.FAILED)
        self.assertEqual(detail, "e1")
        self.assertEqual(miss, ["A", "B"])

    def test_aggregate_no_token_and_no_data(self):
        self.assertEqual(aggregate_synergy_status({"A": (SynergyStatus.NO_TOKEN, "")})[0], SynergyStatus.NO_TOKEN)
        self.assertEqual(aggregate_synergy_status({"A": (SynergyStatus.NO_DATA, ""), "B": (SynergyStatus.NO_DATA, "")})[0],
                         SynergyStatus.NO_DATA)

    def test_aggregate_failed_first_ok_second(self):  # раньше давало FAILED
        st, _, miss = aggregate_synergy_status({"A": (SynergyStatus.FAILED, "e"), "B": (SynergyStatus.OK, "")})
        self.assertEqual((st, miss), (SynergyStatus.PARTIAL_DATA, ["A"]))


# ============================ HTTP-СЛОЙ ============================
class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._p = status, payload
    def json(self): return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

def make_client(side_effects):
    c = OpenDotaClient(api_key="x")
    c.cache = AtomicCache(file_path=os.path.join(tempfile.mkdtemp(), "c.json"))
    c.session = mock.Mock()
    c.session.get.side_effect = side_effects
    return c

@mock.patch("api.time.sleep")
class TestHttp(unittest.TestCase):
    def test_429_three_times(self, _s):
        c = make_client([FakeResp(429)] * 3)
        with self.assertRaises(ApiUnavailableError):
            c._request("/heroes")
        self.assertEqual(c.session.get.call_count, 3)

    def test_429_then_ok(self, _s):
        c = make_client([FakeResp(429), FakeResp(200, [1])])
        self.assertEqual(c._request("/heroes"), [1])

    def test_404_no_retry(self, _s):
        c = make_client([FakeResp(404)])
        with self.assertRaises(ApiUnavailableError):
            c._request("/heroes")
        self.assertEqual(c.session.get.call_count, 1)

    def test_500_twice_then_ok(self, _s):
        c = make_client([FakeResp(500), FakeResp(500), FakeResp(200, [1])])
        self.assertEqual(c._request("/heroes"), [1])
        self.assertEqual(c.session.get.call_count, 3)

    def test_500_always(self, _s):
        c = make_client([FakeResp(500)] * 3)
        with self.assertRaises(ApiUnavailableError):
            c._request("/heroes")

    def test_wrong_type(self, _s):
        c = make_client([FakeResp(200, {"error": "x"})])
        with self.assertRaises(ApiUnavailableError):
            c._request("/heroes", expected_type=list)

    def test_timeouts(self, _s):
        c = make_client([requests.exceptions.Timeout()] * 3)
        with self.assertRaises(ApiUnavailableError):
            c._request("/heroes")
        self.assertEqual(c.session.get.call_count, 3)

    def test_cache_hit_skips_network(self, _s):
        c = make_client([FakeResp(200, [1, 2])])
        self.assertEqual(c._request("/heroes", use_cache=True), [1, 2])
        self.assertEqual(c._request("/heroes", use_cache=True), [1, 2])
        self.assertEqual(c.session.get.call_count, 1)

    def test_matches_filter_turbo_and_limit(self, _s):
        data = [{'lobby_type': 0, 'game_mode': 23, 'hero_id': 1},   # турбо: отбросить
                {'lobby_type': 1, 'game_mode': 22, 'hero_id': 2},   # практика: отбросить
                {'lobby_type': 7, 'game_mode': 22, 'hero_id': 3},
                {'lobby_type': 0, 'game_mode': 1, 'hero_id': 4},
                {'lobby_type': 0, 'game_mode': 22, 'hero_id': 5}]
        c = make_client([FakeResp(200, data)])
        valid, used = c.get_player_matches(1, 2)
        self.assertEqual([m['hero_id'] for m in valid], [3, 4])
        self.assertEqual(used, 2)

    def test_matches_private_and_empty(self, _s):
        with self.assertRaises(PrivateProfileError):
            make_client([FakeResp(200, {"error": "private"})]).get_player_matches(1, 50)
        with self.assertRaises(EmptyHistoryError):
            make_client([FakeResp(200, [])]).get_player_matches(1, 50)

class TestStratzClient(unittest.TestCase):
    def mk(self, token="t"):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = StratzClient(token=token)
        c.session = mock.Mock()
        c.cache = AtomicCache(file_path=os.path.join(tempfile.mkdtemp(), 's.json'))
        return c

    def test_no_token(self):
        self.assertEqual(self.mk(None).get_synergy(1)[0], SynergyStatus.NO_TOKEN)

    def test_401(self):
        c = self.mk(); c.session.post.return_value = FakeResp(401)
        self.assertEqual(c.get_synergy(1)[0], SynergyStatus.FAILED)

    def test_graphql_errors_status_200(self):
        c = self.mk(); c.session.post.return_value = FakeResp(200, {"errors": [{}], "data": None})
        self.assertEqual(c.get_synergy(1)[0], SynergyStatus.FAILED)

    def test_timeout(self):
        c = self.mk(); c.session.post.side_effect = requests.exceptions.Timeout()
        self.assertEqual(c.get_synergy(1)[0], SynergyStatus.FAILED)

    def test_query_building(self):
        q = StratzClient.build_query(1, 150, "DIVINE_IMMORTAL")
        self.assertIn("matchUp(heroId: 1, take: 150, bracketBasicIds: [DIVINE_IMMORTAL])", q)
        self.assertIn("vs { heroId2 matchCount winCount }", q)
        self.assertIn("LEGEND_ANCIENT, DIVINE_IMMORTAL", StratzClient.build_query(2, 10, "LEGEND_ANCIENT, DIVINE_IMMORTAL"))
        with self.assertRaises(ValueError):
            StratzClient.build_query(1, 150, "X]) { evil")

    def test_counters_and_cache(self):
        c = self.mk()
        c.session.post.return_value = FakeResp(200, {"data": {"heroStats": {"matchUp": [{
            "with": [{"heroId2": 5, "matchCount": 10, "winCount": 6}],
            "vs": [{"heroId2": 94, "matchCount": 288, "winCount": 192}]}]}}})
        st, _, vs = c.get_counters(1)
        self.assertEqual(st, SynergyStatus.OK); self.assertEqual(vs[94]["winCount"], 192)
        st, _, w = c.get_synergy(1)                       # второй вызов берётся из кэша
        self.assertEqual(st, SynergyStatus.OK); self.assertEqual(w[5]["matchCount"], 10)
        self.assertEqual(c.session.post.call_count, 1)

    def test_failed_response_not_cached(self):
        c = self.mk()
        c.session.post.return_value = FakeResp(200, {"errors": [{"message": "bad"}], "data": None})
        c.get_counters(1); c.get_counters(1)
        self.assertEqual(c.session.post.call_count, 2)

    def test_ok(self):
        c = self.mk()
        c.session.post.return_value = FakeResp(200, {"data": {"heroStats": {"matchUp": {"with": [{"heroId2": 5}]}}}})
        st, _, data = c.get_synergy(1)
        self.assertEqual(st, SynergyStatus.OK); self.assertIn(5, data)


# ============================ КОНВЕЙЕР (фейковые клиенты) ============================
class FakeCache:
    def __init__(self): self.flushes = 0
    def flush(self): self.flushes += 1

class FakeOD:
    def __init__(self, heroes, matches=(), matchups=None, error=None, base=None, base_high=None):
        self.heroes, self.matches, self.matchups, self.error = heroes, list(matches), matchups or {}, error
        self.base, self.base_high = base or {}, base_high or {}
        self.cache = FakeCache()
    def get_heroes(self): return {h['id']: h for h in self.heroes}
    def get_hero_stats(self):
        out = []
        for h in self.heroes:
            row = {'id': h['id'], '1_pick': 1000, '1_win': int(self.base.get(h['id'], 0.5) * 1000)}
            if h['id'] in self.base_high:     # высокие бракеты: 7 и 8 по 500 пиков
                row.update({'7_pick': 500, '7_win': int(self.base_high[h['id']] * 500),
                            '8_pick': 500, '8_win': int(self.base_high[h['id']] * 500)})
            out.append(row)
        return out
    def get_matchups(self, hid): return self.matchups.get(hid, [])
    def get_player_matches(self, acc, limit):
        if self.error: raise self.error
        return self.matches, len(self.matches)

class FakeStratz:
    def __init__(self, responses=None, token="t", counters=None):
        self.responses, self.token, self.counters = responses or {}, token, counters or {}
        self.cache = FakeCache()
    def get_counters(self, enemy_id):
        if not self.token: return SynergyStatus.NO_TOKEN, "Token not provided", {}
        return self.counters.get(enemy_id, (SynergyStatus.NO_DATA, "none", {}))
    def get_synergy(self, ally_id):
        if not self.token: return SynergyStatus.NO_TOKEN, "Token not provided", {}
        return self.responses.get(ally_id, (SynergyStatus.NO_DATA, "none", {}))

def hero(i, name): return {'id': i, 'name': f'npc_dota_hero_{name}', 'localized_name': name.upper()}

HEROES = [hero(1, 'a'), hero(2, 'b'), hero(3, 'c'), hero(4, 'd'), hero(10, 'e'), hero(11, 'ally1'), hero(12, 'ally2')]
DB = {1: ["A", "A", [1]], 2: ["B", "B", [1]], 3: ["C", "C", [1]], 4: ["D", "D", [2]],
      10: ["E", "E", [1]], 11: ["Ally1", "Ally1", [1, 5]], 12: ["Ally2", "Ally2", [1, 5]]}

def enemy_matchups(enemy_wins_by_hero, games=1000):
    return [{'hero_id': h, 'games_played': games, 'wins': w} for h, w in enemy_wins_by_hero.items()]

def b_matches(n=20, wins=14):
    t = __import__('time').time() - 86400
    return [{'hero_id': 2, 'start_time': t, 'player_slot': 0, 'radiant_win': i < wins,
             'kills': 8, 'deaths': 2, 'assists': 10} for i in range(n)]

class TestPipeline(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(analysis, "HERO_DB", DB), mock.patch.object(analysis, "HERO_ALIASES", {})):
            p.start(); self.addCleanup(p.stop)

    def run_rec(self, od, stratz=None, role=1, enemies=(), allies=(), only_played=False, min_games=3):
        an = DotaAnalyzer(od, stratz or FakeStratz())
        en = [an.heroes[i] for i in enemies]; al = [an.heroes[i] for i in allies]
        return an, an.get_recommendations(1, role, en, al, 100, min_games, only_played)

    def test_counter_beats_form_when_strong(self):
        od = FakeOD(HEROES, b_matches(), {10: enemy_matchups({1: 400, 2: 500, 3: 500, 4: 500})})
        _, res = self.run_rec(od, enemies=[10])
        order = [c.hero.id for c in res.candidates]
        self.assertEqual(order[0], 1)   # +4 п.п. контрпика, без практики
        self.assertEqual(order[1], 2)   # сильная форма, нейтральный матчап
        self.assertEqual(order[2], 3)
        a, b = res.candidates[0], res.candidates[1]
        print("\n  A:", round(a.form_score, 3), round(a.draft_score, 3), round(a.final_score, 3),
              "| B:", round(b.form_score, 3), round(b.draft_score, 3), round(b.final_score, 3))
        adv = smooth_winrate(600, 1000, MATCHUP_PRIOR_GAMES, 0.5) - 0.5   # сырые +10 п.п. -> после сглаживания ~+7.7
        self.assertAlmostEqual(a.final_score, 0.4 * 0.5 + 0.6 * sigmoid_scaled(adv, COUNTER_SCALE), places=3)
        self.assertAlmostEqual(a.expected_wr, 0.5 + adv, places=3)

    def test_form_beats_weak_counter(self):
        od = FakeOD(HEROES, b_matches(), {10: enemy_matchups({1: 490, 2: 500, 3: 500, 4: 500})})
        _, res = self.run_rec(od, enemies=[10])
        self.assertEqual(res.candidates[0].hero.id, 2)   # +1 п.п. не перебивает форму

    def test_role_filter_and_exclusions(self):
        od = FakeOD(HEROES, b_matches(), {10: enemy_matchups({})})
        _, res = self.run_rec(od, enemies=[10], allies=[11])
        ids = {c.hero.id for c in res.candidates}
        self.assertNotIn(4, ids)                  # роль 2
        self.assertNotIn(10, ids)                 # враг
        self.assertNotIn(11, ids)                 # союзник (роль 1/5, но выбран)
        self.assertEqual(ids, {1, 2, 3, 12})

    def test_only_played_min_games(self):
        t = __import__('time').time() - 86400
        a2 = [{'hero_id': 1, 'start_time': t, 'player_slot': 0, 'radiant_win': True, 'kills': 5, 'deaths': 5, 'assists': 5}] * 2
        od = FakeOD(HEROES, b_matches() + a2)
        _, res = self.run_rec(od, only_played=True, min_games=3)
        self.assertEqual([c.hero.id for c in res.candidates], [2])   # у A всего 2 игры

    def test_only_played_drops_bad_form(self):
        tnow = __import__('time').time() - 86400
        bad = [{'hero_id': 3, 'start_time': tnow, 'player_slot': 0, 'radiant_win': i == 0,
                'kills': 1, 'deaths': 9, 'assists': 2} for i in range(10)]   # 1 победа из 10, KDA 0.3
        od = FakeOD(HEROES, b_matches() + bad)
        _, res = self.run_rec(od, only_played=True)
        self.assertEqual([c.hero.id for c in res.candidates], [2])          # плохая форма отсеяна
        _, res = self.run_rec(od, only_played=False)
        self.assertIn(3, [c.hero.id for c in res.candidates])               # без фильтра герой остаётся

    def test_small_sample_warning(self):
        od = FakeOD(HEROES, b_matches(n=4, wins=3))
        _, res = self.run_rec(od)
        b = next(c for c in res.candidates if c.hero.id == 2)
        self.assertTrue(any("Мало игр" in w for w in b.warnings))

    def test_matches_used_warning(self):
        od = FakeOD(HEROES, b_matches(n=10))
        _, res = self.run_rec(od)
        self.assertEqual((res.matches_used, res.matches_requested), (10, 100))
        self.assertTrue(any("10 из 100" in w for w in res.warnings))

    def test_small_sample_does_not_win(self):
        # +11 п.п. сырых на 36 играх (как Лон Друид против Бристлбека на скриншоте): после сглаживания <2 п.п.
        od = FakeOD(HEROES, b_matches(n=20, wins=10), {10: enemy_matchups({1: 400 // 28 * 1, 2: 500, 3: 500, 4: 500}, games=36)})
        od.matchups[10] = [{'hero_id': 1, 'games_played': 36, 'wins': 14},      # кандидат A: сырой WR 61% (+11)
                           {'hero_id': 2, 'games_played': 5000, 'wins': 2450},  # B: 51% на 5000 играх (+1)
                           {'hero_id': 3, 'games_played': 5000, 'wins': 2500}]
        _, res = self.run_rec(od, enemies=[10])
        a = next(c for c in res.candidates if c.hero.id == 1)
        b = next(c for c in res.candidates if c.hero.id == 2)
        self.assertLess(a.matchups[0].advantage, 0.02)       # шум не раздувает оценку
        self.assertGreater(b.matchups[0].advantage, a.matchups[0].advantage * 0.5)
        self.assertAlmostEqual(a.matchups[0].raw_wr, 1 - 14 / 36)
        self.assertTrue(a.matchups[0].warnings)               # предупреждение «мало игр»

    def test_base_strength_matters(self):
        # A: слабый в среднем (46%), в матчапе 49% (+3 п.п. к его среднему); B: 53% и без поправки.
        # Ожидаемый WR A = ~49%, B = 53% -> B должен быть выше, хотя «преимущество» у A больше.
        od = FakeOD(HEROES, (), {10: [{'hero_id': 1, 'games_played': 5000, 'wins': 2550},
                                       {'hero_id': 2, 'games_played': 5000, 'wins': 2350},
                                       {'hero_id': 3, 'games_played': 5000, 'wins': 2500}]},
                    base={1: 0.46, 2: 0.53, 3: 0.50})
        _, res = self.run_rec(od, enemies=[10])
        order = [c.hero.id for c in res.candidates]
        self.assertLess(order.index(2), order.index(1))
        a = next(c for c in res.candidates if c.hero.id == 1)
        b = next(c for c in res.candidates if c.hero.id == 2)
        self.assertGreater(a.matchups[0].advantage, b.matchups[0].advantage - 0.001)   # у A поправка не меньше
        self.assertLess(a.expected_wr, b.expected_wr)
        self.assertAlmostEqual(a.base_wr, 0.46)

    def test_expected_wr_none_without_enemies(self):
        _, res = self.run_rec(FakeOD(HEROES, b_matches()))
        self.assertTrue(all(c.expected_wr is None for c in res.candidates))

    def stratz_vs(self, wins_by_hero, games=2000):
        return {h: {'heroId2': h, 'matchCount': games, 'winCount': w} for h, w in wins_by_hero.items()}

    def test_stratz_counters_used_with_high_base(self):
        # Враг E (id 10) по данным Stratz выигрывает у A всего 40% -> A выигрывает 60% против E
        stratz = FakeStratz(counters={10: (SynergyStatus.OK, "Success", self.stratz_vs({1: 800, 2: 1000, 3: 1000}))})
        od = FakeOD(HEROES, b_matches(), {10: enemy_matchups({1: 500, 2: 500, 3: 500})},   # OpenDota здесь нейтрален
                    base={1: 0.50, 2: 0.50, 3: 0.50}, base_high={1: 0.52, 2: 0.52, 3: 0.52})
        _, res = self.run_rec(od, stratz, enemies=[10])
        self.assertTrue(res.counter_source.startswith("Stratz"))
        a = next(c for c in res.candidates if c.hero.id == 1)
        self.assertAlmostEqual(a.matchups[0].raw_wr, 0.60)
        self.assertAlmostEqual(a.base_wr, 0.52, places=2)            # база из высоких бракетов
        self.assertGreater(a.matchups[0].advantage, 0.0)
        self.assertEqual(res.candidates[0].hero.id, 1)               # контрпик по Stratz поднял A

    def test_stratz_failure_falls_back_to_opendota(self):
        stratz = FakeStratz(counters={10: (SynergyStatus.FAILED, "timeout", {})})
        od = FakeOD(HEROES, b_matches(), {10: enemy_matchups({1: 400, 2: 500, 3: 500})})
        _, res = self.run_rec(od, stratz, enemies=[10])
        self.assertTrue(res.counter_source.startswith("OpenDota"))
        self.assertTrue(any("недоступны" in w and "timeout" in w for w in res.warnings))
        self.assertGreater(res.candidates[0].matchups[0].raw_wr, 0.5)    # данные OpenDota реально использованы

    def test_no_token_uses_opendota_with_note(self):
        _, res = self.run_rec(FakeOD(HEROES, b_matches(), {10: enemy_matchups({1: 500})}), FakeStratz(token=None), enemies=[10])
        self.assertTrue(res.counter_source.startswith("OpenDota"))
        self.assertTrue(any("Нет токена Stratz" in w for w in res.warnings))

    # ----- синергия -----
    SYN_OK = {1: {'heroId2': 1, 'matchCount': 1000, 'winCount': 560}}

    def test_synergy_partial(self):
        stratz = FakeStratz({11: (SynergyStatus.OK, "Success", self.SYN_OK), 12: (SynergyStatus.FAILED, "timeout", {})})
        _, res = self.run_rec(FakeOD(HEROES, b_matches()), stratz, allies=[11, 12])
        # ally1 и ally2 выбраны -> не кандидаты; кандидаты A, B, C
        self.assertEqual(res.synergy_status, SynergyStatus.PARTIAL_DATA)
        self.assertEqual(res.synergy_missing_allies, ["Ally2"])
        self.assertTrue(any("неполные" in w for w in res.warnings))
        a = next(c for c in res.candidates if c.hero.id == 1)
        b = next(c for c in res.candidates if c.hero.id == 2)
        self.assertEqual([s.ally_name for s in a.synergies], ["Ally1"])
        self.assertEqual(a.synergy_missing, ["Ally2"])
        self.assertGreater(a.draft_score, 0.65)     # синергия сырые +6 п.п. -> после сглаживания ~+4.6 -> ~0.72
        self.assertEqual(b.synergies, [])
        self.assertAlmostEqual(b.draft_score, 0.5)  # нет данных -> нейтральный драфт, не штраф
        self.assertEqual(b.synergy_missing, ["Ally1", "Ally2"])

    def test_synergy_no_token_and_failed(self):
        _, res = self.run_rec(FakeOD(HEROES, b_matches()), FakeStratz(token=None), allies=[11])
        self.assertEqual(res.synergy_status, SynergyStatus.NO_TOKEN)
        stratz = FakeStratz({11: (SynergyStatus.FAILED, "boom", {})})
        _, res = self.run_rec(FakeOD(HEROES, b_matches()), stratz, allies=[11])
        self.assertEqual((res.synergy_status, res.synergy_detail), (SynergyStatus.FAILED, "boom"))
        self.assertTrue(all(c.draft_score == 0.5 for c in res.candidates))

    def test_synergy_with_enemies_weights(self):
        stratz = FakeStratz({11: (SynergyStatus.OK, "Success", self.SYN_OK)})
        od = FakeOD(HEROES, b_matches(), {10: enemy_matchups({1: 500, 2: 500, 3: 500})})
        _, res = self.run_rec(od, stratz, enemies=[10], allies=[11])
        a = next(c for c in res.candidates if c.hero.id == 1)
        wc, ws = 0.7 * (1 / 5), 0.3 * (1 / 4)      # известен 1 враг из 5 и 1 союзник из 4
        self.assertAlmostEqual(a.draft_score, (wc * a.counter_score + ws * a.synergy_score) / (wc + ws))

    def test_incomplete_enemy_draft_warning_and_weights(self):
        od = FakeOD(HEROES, b_matches(), {10: enemy_matchups({1: 500})})
        _, res = self.run_rec(od, enemies=[10])
        self.assertTrue(any("неполный" in w for w in res.warnings))
        # больше врагов известно -> предупреждения нет
        heroes = HEROES + [hero(20, 'e2'), hero(21, 'e3'), hero(22, 'e4'), hero(23, 'e5')]
        db = {**DB, **{i: [f"E{i}", f"E{i}", [1]] for i in (20, 21, 22, 23)}}
        with mock.patch.object(analysis, "HERO_DB", db):
            an = DotaAnalyzer(FakeOD(heroes, b_matches()), FakeStratz())
            res = an.get_recommendations(1, 1, [an.heroes[i] for i in (10, 20, 21, 22)], [], 100, 3, False)
        self.assertFalse(any("неполный" in w for w in res.warnings))

    def test_more_synergy_weight_when_few_enemies_known(self):
        stratz = FakeStratz({11: (SynergyStatus.OK, "ok", {1: {'heroId2': 1, 'matchCount': 2000, 'winCount': 1200}})})
        od = FakeOD(HEROES, b_matches(), {10: enemy_matchups({1: 1000}, games=2000)})
        _, res = self.run_rec(od, stratz, enemies=[10], allies=[11])
        a = next(c for c in res.candidates if c.hero.id == 1)
        # 1 враг (0.2) и 1 союзник (0.25): синергия весит больше контрпика, несмотря на базовое 0.7/0.3
        wc, ws = 0.7 * 0.2, 0.3 * 0.25
        self.assertGreater(ws / (wc + ws), 0.3)
        self.assertAlmostEqual(a.draft_score, (wc * a.counter_score + ws * a.synergy_score) / (wc + ws))

    def test_ally_roles_change_synergy_weight(self):
        ok = lambda w: (SynergyStatus.OK, "ok", {1: {'heroId2': 1, 'matchCount': 2000, 'winCount': w}})
        stratz = FakeStratz({11: ok(1400), 12: ok(600)})            # 11 сильная связка (+), 12 слабая (-)
        an = DotaAnalyzer(FakeOD(HEROES, b_matches()), stratz)
        def score(roles):
            res = an.get_recommendations(1, 1, [], [an.heroes[11], an.heroes[12]], 100, 3, False, roles)
            return next(c for c in res.candidates if c.hero.id == 1).synergy_score
        support_on_good = score({11: 5, 12: 3})     # керри + саппорт с сильной связкой: она весит 1.5x
        support_on_bad = score({11: 3, 12: 5})
        self.assertGreater(support_on_good, support_on_bad)

    # ----- ошибки и кэш -----
    def test_errors_propagate_and_flush(self):
        for exc in (PrivateProfileError("x"), EmptyHistoryError("x"), ApiUnavailableError("x")):
            od = FakeOD(HEROES, error=exc)
            an = DotaAnalyzer(od, FakeStratz())
            before = od.cache.flushes
            with self.assertRaises(type(exc)):
                an.get_recommendations(1, 1, [], [], 100, 3, False)
            self.assertGreater(od.cache.flushes, before)

    def test_init_flushes_cache(self):
        od = FakeOD(HEROES)
        DotaAnalyzer(od, FakeStratz())
        self.assertGreaterEqual(od.cache.flushes, 1)


# ============================ ВЫДАЧА 2 + 1 ============================
def cr(i, games, form, draft, final=None, base=0.5):
    return SimpleNamespace(hero=SimpleNamespace(id=i), form_games=games, form_score=form, draft_score=draft,
                           final_score=final if final is not None else 0.4 * form + 0.6 * draft, base_wr=base)

class TestSelection(unittest.TestCase):
    def test_two_pool_one_meta(self):
        cands = [cr(1, 20, 0.7, 0.5), cr(2, 10, 0.6, 0.6), cr(3, 8, 0.55, 0.4),   # пул
                 cr(4, 0, 0.5, 0.9),                                             # без практики, но отличный против драфта
                 cr(5, 0, 0.5, 0.8)]
        picks = select_recommendations(cands)
        self.assertEqual([(k, c.hero.id) for k, c in picks], [("pool", 2), ("pool", 1), ("meta", 4)])

    def test_pool_requires_games_and_form(self):
        cands = [cr(1, 2, 0.9, 0.5), cr(2, 10, 0.3, 0.5), cr(3, 0, 0.5, 0.7)]   # мало игр / плохая форма / нет практики
        self.assertEqual([k for k, _ in select_recommendations(cands)], ["meta", "meta", "meta"])

    def test_short_pool_filled_by_meta(self):
        cands = [cr(1, 10, 0.6, 0.5), cr(2, 0, 0.5, 0.8), cr(3, 0, 0.5, 0.7), cr(4, 0, 0.5, 0.6)]
        picks = select_recommendations(cands)
        self.assertEqual([(k, c.hero.id) for k, c in picks], [("pool", 1), ("meta", 2), ("meta", 3)])

    def test_meta_ties_broken_by_base_wr(self):
        cands = [cr(1, 0, 0.5, 0.5, base=0.48), cr(2, 0, 0.5, 0.5, base=0.54)]
        self.assertEqual(select_recommendations(cands, n_pool=0, n_total=1)[0][1].hero.id, 2)

    def test_meta_pick_not_from_pool_slots(self):
        cands = [cr(1, 10, 0.7, 0.9), cr(2, 10, 0.6, 0.8), cr(3, 10, 0.55, 0.7)]
        picks = select_recommendations(cands)
        self.assertEqual([c.hero.id for _, c in picks], [1, 2, 3])
        self.assertEqual([k for k, _ in picks], ["pool", "pool", "meta"])      # третий лучший по драфту из остатка

# ============================ CLI: строка про синергию ============================
class TestRoleWeight(unittest.TestCase):
    def test_weights(self):
        self.assertEqual(synergy_role_weight(1, 5), 1.5); self.assertEqual(synergy_role_weight(2, 4), 1.5)
        self.assertEqual(synergy_role_weight(5, 1), 1.5); self.assertEqual(synergy_role_weight(4, 2), 1.5)
        self.assertEqual(synergy_role_weight(1, 3), 1.0); self.assertEqual(synergy_role_weight(3, 5), 1.0)
        self.assertEqual(synergy_role_weight(1, None), 1.0); self.assertEqual(synergy_role_weight(1, 2), 1.0)

class TestSynergyLine(unittest.TestCase):
    def cand(self, syn=(), missing=()):
        return SimpleNamespace(synergies=[SimpleNamespace(ally_name=n, advantage=a, games=g) for n, a, g in syn],
                               synergy_missing=list(missing))
    def res(self, st, detail=""): return SimpleNamespace(synergy_status=st, synergy_detail=detail)

    def test_no_allies_no_line(self):
        self.assertIsNone(cli.synergy_line(self.cand(), self.res(SynergyStatus.OK), False))

    def test_values(self):
        s = cli.synergy_line(self.cand([("X", 0.05, 900)]), self.res(SynergyStatus.OK), True)
        self.assertIn("+5.0 п.п.", s); self.assertIn("900", s)

    def test_partial_lists_missing(self):
        s = cli.synergy_line(self.cand([("X", 0.02, 100)], ["Y"]), self.res(SynergyStatus.PARTIAL_DATA), True)
        self.assertIn("нет данных по: Y", s)

    def test_no_data_never_silent(self):
        s = cli.synergy_line(self.cand([], ["X", "Y"]), self.res(SynergyStatus.NO_DATA), True)
        self.assertIn("нет данных по: X, Y", s)

    def test_disabled(self):
        s = cli.synergy_line(self.cand([], ["X"]), self.res(SynergyStatus.FAILED, "timeout"), True)
        self.assertIn("отключена", s); self.assertIn("timeout", s)
        s = cli.synergy_line(self.cand([], ["X"]), self.res(SynergyStatus.NO_TOKEN, "нет токена STRATZ_TOKEN"), True)
        self.assertIn("отключена", s)


# ============================ КАЛИБРОВКА ВКЛАДОВ ============================
class MockCand:
    def __init__(self, f, d, advs):
        self.form_score, self.draft_score = f, d
        self.matchups = [SimpleNamespace(advantage=a) for a in advs]

def make_pool(seed, n_played=5, adv_sd=0.02, n_enemies=5):
    """25 неигранных (форма 0.5) + 5 сыгранных; драфт = sigmoid_scaled(среднее преимущество по 5 врагам)."""
    rnd = random.Random(seed)
    def draft():
        advs = [rnd.gauss(0, adv_sd) for _ in range(n_enemies)]
        return sigmoid_scaled(sum(advs) / n_enemies, COUNTER_SCALE), advs
    pool = []
    for _ in range(25):
        d, advs = draft(); pool.append(MockCand(0.5, d, advs))
    for _ in range(n_played):
        d, advs = draft(); pool.append(MockCand(0.5 + rnd.gauss(0, 0.1), d, advs))
    return pool

class TestContribution(unittest.TestCase):
    def test_report_arithmetic(self):
        c = [MockCand(0.4, 0.5, []), MockCand(0.6, 0.5, [])]
        sf, sd, ff, fd, avg, mx = contribution_report(c)
        self.assertAlmostEqual(sf, 0.1); self.assertEqual(sd, 0.0)
        self.assertAlmostEqual(ff, 1.0); self.assertAlmostEqual(fd, 0.0)

    def test_realistic_pool_shares(self):
        # КАЛИБРОВКА ПО СИНТЕТИКЕ, не по реальным данным: решающее слово за --debug на вашем ID.
        for adv_sd in (0.01, 0.02, 0.03):
            sf, sd, ff, fd, _, _ = contribution_report(make_pool(7, adv_sd=adv_sd))
            print(f"\n  adv_sd={adv_sd}: std_f={sf:.3f} std_d={sd:.3f} форма={ff*100:.0f}% драфт={fd*100:.0f}%")
        sf, sd, ff, fd, _, _ = contribution_report(make_pool(7, adv_sd=0.02))
        self.assertGreater(ff, 0.0)
        self.assertAlmostEqual(ff + fd, 1.0)

if __name__ == '__main__':
    unittest.main()
