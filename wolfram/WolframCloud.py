from urllib import urlencode
from urllib2 import urlopen

class WolframCloud:

    def wolfram_cloud_call(self, **args):
        arguments = dict([(key, arg) for key, arg in args.iteritems()])
        result = urlopen("http://www.wolframcloud.com/objects/91a10361-47b0-406d-89aa-3117f4d46d0c", urlencode(arguments))
        return result.read()

    def call(self, x):
        textresult =  self.wolfram_cloud_call(x=x)
        return textresult
