from urllib import urlencode
from urllib2 import urlopen

class WolframCloud:

    def wolfram_cloud_call(self, **args):
        arguments = dict([(key, arg) for key, arg in args.iteritems()])
        result = urlopen("http://www.wolframcloud.com/objects/76f48863-8a68-4d2d-8698-ee2b3ac88aa0", urlencode(arguments))
        return result.read()

    def call(self, x):
        textresult =  self.wolfram_cloud_call(x=x)
        return textresult
