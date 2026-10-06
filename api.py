# api.py
import os
import json
import time
import requests
import logging
import re
from config import (STRATZ_MATCHUP_QUERY, STRATZ_BRACKETS, STRATZ_TAKE, STRATZ_TIMEOUT, CACHE_TTL)
from utils import SynergyStatus, parse_matchup_response

logger = logging.getLogger(__name__)

class ApiUnavailableError(Exception): pass
class PrivateProfileError(Exception): pass
class EmptyHistoryError(Exception): pass

class AtomicCache:
    def __init__(self, file_path: str = "dota_cache.json", ttl: int = CACHE_TTL):
        self.file_path = file_path
        self.ttl = ttl
        self.data = self._load()
        self.dirty = False

    def _load(self) -> dict:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception: pass
        return {}

    def get(self, key: str):
        item = self.data.get(key)
        if item and time.time() - item['timestamp'] < self.ttl:
            return item['value']
        return None

    def set(self, key: str, value):
        self.data[key] = {'timestamp': time.time(), 'value': value}
        self.dirty = True

    def flush(self):
        if not self.dirty: return
        tmp_file = f"{self.file_path}.tmp"
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False)
            os.replace(tmp_file, self.file_path)
            self.dirty = False
        except Exception as e:
            logger.warning(f"Ошибка сохранения кэша: {e}")

class OpenDotaClient:
    BASE_URL = "https://api.opendota.com/api"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENDOTA_API_KEY")
        self.session = requests.Session()
        self.cache = AtomicCache()

    def _request(self, endpoint: str, params: dict = None, use_cache: bool = False, expected_type=None) -> any:
        if use_cache:
            cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True) if params else ''}"
            cached = self.cache.get(cache_key)
            if cached is not None: return cached

        url = f"{self.BASE_URL}{endpoint}"
        req_params = {'api_key': self.api_key} if self.api_key else {}
        if params: req_params.update(params)

        retries = 3
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=req_params, timeout=10)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()

                if expected_type and not isinstance(data, expected_type):
                    raise ApiUnavailableError(f"Неожиданный формат ответа {endpoint}. Ожидался {expected_type.__name__}.")

                if use_cache:
                    cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True) if params else ''}"
                    self.cache.set(cache_key, data)
                return data

            except requests.exceptions.HTTPError as e:
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise ApiUnavailableError(f"HTTP {e.response.status_code} для {url}") from e
                if attempt == retries - 1: raise ApiUnavailableError(f"Ошибка API: {e}") from e
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1: raise ApiUnavailableError(f"Сетевая ошибка: {e}") from e
                time.sleep(2 ** attempt)
        raise ApiUnavailableError("Превышено число попыток OpenDota.")

    def get_heroes(self) -> dict:
        data = self._request("/heroes", use_cache=True, expected_type=list)
        return {h['id']: h for h in data}

    def get_hero_stats(self) -> dict:
        return self._request("/heroStats", use_cache=True, expected_type=list)

    def get_matchups(self, hero_id: int) -> list:
        return self._request(f"/heroes/{hero_id}/matchups", use_cache=True, expected_type=list)

    def get_player_matches(self, account_id: int, target_limit: int) -> tuple[list, int]:
        params = {
            "limit": 200,
            "project": ["hero_id", "player_slot", "radiant_win", "kills", "deaths", "assists", "start_time", "lobby_type", "game_mode"]
        }
        data = self._request(f"/players/{account_id}/matches", params=params, use_cache=False, expected_type=None)

        if isinstance(data, dict) and data.get("error"):
            raise PrivateProfileError("Профиль скрыт.")
        if not data or not isinstance(data, list):
            raise EmptyHistoryError("Пустая история матчей.")

        valid = []
        for m in data:
            if m.get('lobby_type') in (0, 7) and m.get('game_mode') in (1, 2, 3, 4, 5, 16, 22):
                valid.append(m)
            if len(valid) == target_limit:
                break
        return valid, len(valid)

class StratzClient:
    BASE_URL = "https://api.stratz.com/graphql"

    def __init__(self, token: str = None):
        self.token = token or os.environ.get("STRATZ_TOKEN")
        self.session = requests.Session()
        self.cache = AtomicCache(file_path="stratz_cache.json")

    @staticmethod
    def build_query(hero_id: int, take: int = STRATZ_TAKE, brackets: str = STRATZ_BRACKETS) -> str:
        """Запрос строится подстановкой (heroId приводится к int, бракеты проверяются): без переменных GraphQL."""
        if not re.fullmatch(r"[A-Z_]+(\s*,\s*[A-Z_]+)*", brackets or ""):
            raise ValueError(f"Некорректные ранги Stratz: {brackets!r}")
        return STRATZ_MATCHUP_QUERY % {"hero_id": int(hero_id), "take": int(take), "brackets": brackets}

    def get_matchup_data(self, hero_id: int):
        """(статус, пояснение, with{}, vs{}) для героя. Успешные ответы кэшируются на 24 часа."""
        if not self.token:
            return SynergyStatus.NO_TOKEN, "Token not provided", {}, {}
        key = f"stratz_matchup_{int(hero_id)}_{STRATZ_BRACKETS}_{STRATZ_TAKE}"
        cached = self.cache.get(key)
        if cached is not None:
            return parse_matchup_response(cached)
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "STRATZ_API"}
        try:
            resp = self.session.post(self.BASE_URL, json={"query": self.build_query(hero_id)},
                                     headers=headers, timeout=STRATZ_TIMEOUT)
            if resp.status_code in (401, 403):
                return SynergyStatus.FAILED, "401/403 Invalid token or Cloudflare block", {}, {}
            resp.raise_for_status()
            payload = resp.json()
            result = parse_matchup_response(payload)
            if result[0] == SynergyStatus.OK:
                self.cache.set(key, payload)
            return result
        except Exception as e:
            return SynergyStatus.FAILED, f"Network/Timeout: {e}", {}, {}

    def get_synergy(self, ally_id: int):
        """(статус, пояснение, with{}) — синергия с союзником."""
        st, detail, with_d, _ = self.get_matchup_data(ally_id)
        if st == SynergyStatus.OK and not with_d:
            return SynergyStatus.NO_DATA, "Empty 'with' array", {}
        return st, detail, with_d

    def get_counters(self, enemy_id: int):
        """(статус, пояснение, vs{}) — как враг играет против каждого героя (winCount = победы врага)."""
        st, detail, _, vs_d = self.get_matchup_data(enemy_id)
        if st == SynergyStatus.OK and not vs_d:
            return SynergyStatus.NO_DATA, "Empty 'vs' array", {}
        return st, detail, vs_d
