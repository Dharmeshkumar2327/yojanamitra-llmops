import numpy as np

from yojanamitra.api.ratelimit import RateLimiter
from yojanamitra.cache.semantic_cache import InMemoryBackend, SemanticCache


def test_semantic_cache_hit_and_miss():
    cache = SemanticCache(InMemoryBackend(100), threshold=0.95, ttl_seconds=3600)
    v = np.array([1.0, 0.0, 0.0])
    cache.store(v, ["pm_kisan"], 3, {"answer": "Rs 6000"})
    hit = cache.lookup(np.array([0.99, 0.05, 0.0]), ["pm_kisan"], 3)
    assert hit and hit.answer["answer"] == "Rs 6000"
    assert cache.lookup(np.array([0.0, 1.0, 0.0]), ["pm_kisan"], 3) is None  # different meaning
    assert cache.lookup(v, ["pm_jay"], 3) is None  # different scheme filter
    assert cache.lookup(v, ["pm_kisan"], 2) is None  # different prompt version
    assert cache.hits == 1 and cache.misses == 3


def test_semantic_cache_ttl():
    cache = SemanticCache(InMemoryBackend(100), threshold=0.9, ttl_seconds=0)
    cache.store(np.array([1.0, 0.0]), None, 1, {"answer": "x"})
    assert cache.lookup(np.array([1.0, 0.0]), None, 1) is None


def test_rate_limiter():
    rl = RateLimiter(max_requests=2, window_seconds=60)
    assert rl.allow("ip", now=0) and rl.allow("ip", now=1)
    assert not rl.allow("ip", now=2)
    assert rl.allow("other", now=2)
    assert rl.allow("ip", now=62)
