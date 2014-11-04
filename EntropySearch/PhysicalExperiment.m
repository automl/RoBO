function f = PhysicalExperiment(x)

display(['The optimizer proposes x = ']);
display([num2str(x)]);
display(['as the next location to test at. Please perform an experiment with this setting.']);
f = input('please enter the resulting function value now (as a floating point number).\n');