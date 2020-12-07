import lsmtool

s = lsmtool.load('test_apparent_sky.txt')

s.group('meanshift', byPatch=True,lookDistance=0.075,groupingDistance=0.01)

if (len(s.getPatchPositions()) == 67):
    print("Test successful.")
else:
    print("Test failed. (Did you build with c++ extensions? This test may fail for the pure python code.)")


