import lsmtool

s = lsmtool.load('test_large.sky')

s.group('meanshift', byPatch=True,lookDistance=0.075,groupingDistance=0.01)

