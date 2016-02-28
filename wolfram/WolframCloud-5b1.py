from urllib import urlencode
from urllib2 import urlopen

# Remove BackGround Wolfram API

class WolframCloud:

    def wolfram_cloud_call(self, **args):
        arguments = dict([(key, arg) for key, arg in args.iteritems()])
        result = urlopen("http://www.wolframcloud.com/objects/e3c19a70-cf31-46b3-96d6-88e9d5ecba39", urlencode(arguments))
        return result.read()

    def call(self, x):
        textresult =  self.wolfram_cloud_call(x=x)
        return textresult

if __name__ == '__main__':
    ex = WolframCloud()
    mystring = "https://raw.githubusercontent.com/tjohnson314/GraphReader/master/Images/images/graph24.png";
    wow = ex.call(mystring)
    print(wow)