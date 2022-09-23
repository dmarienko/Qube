from collections import namedtuple

# We use this Bar structure as interface only but not as internal holder
__Bar = namedtuple('Bar', ['time', 'open', 'high', 'low', 'close', 'volume'])


class Bar(__Bar):

    def __repr__(self):
        return "[%s]  {o:%f | h:%f | l:%f | c:%f | v:%f}" % (self.time.strftime('%Y-%m-%d %H:%M:%S'),
                                                             self.open, self.high, self.low, self.close, self.volume)
