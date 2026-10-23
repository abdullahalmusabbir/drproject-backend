from rest_framework.throttling import SimpleRateThrottle


class TenSecondThrottle(SimpleRateThrottle):
    scope = 'ten_second'

    def parse_rate(self, rate):
        # 1 request প্রতি 10 সেকেন্ডে, প্রতিটা client (IP) এর জন্য আলাদা
        return (1, 10)

    def get_cache_key(self, request, view):
        ident = self.get_ident(request)
        return self.cache_format % {'scope': self.scope, 'ident': ident}