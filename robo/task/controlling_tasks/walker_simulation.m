function output = walker_simulation(args)
% filename: walker_simulation
% purpose: This file contains all the setup necessary to run a simulation of
%          the three-link biped as a RoBo task.
% The simulator which is called here is a software held by Professor Jessy Grizzle, University of Michigan, 
% Professor Eric Westervelt, The Ohio State University, and Mr. Ben Morris, University of Michigan.

	global t_2 torque y force

	torque = [];
	t_2 = [];
	y = [];
	force = [];

	tstart = 0;
	tfinal = 13;

	% controller parameters
	a1 = args.arg1;
	a2 = args.arg2;
	a3 = args.arg3;
	a4 = args.arg4;
	a5 = args.arg5;
	a6 = args.arg6;
	a7 = args.arg7;
	a8 = args.arg8;

	a = [a1 a2 a3 a4 a5 a6 a7 a8];


	omega_1 = 1.55;
	x0 = sigma_three_link(omega_1,a);
	x0 = transition_three_link(x0).';
	x0 = x0(1:6);

	options = odeset('Events','on','Refine',4,'RelTol',10^-5,'AbsTol',10^-6);

	tout = tstart;
	xout = x0.';
	teout = []; xeout = []; ieout = [];

	steps = args.arg9;
	for i = 1:steps % specified steps to run
		% Solve until the first terminal event.
		[t,x,te,xe,ie] = ode45('walker_main',[tstart tfinal],x0,options,a);

		% Accumulate output.  tout and xout are passed out as output arguments
		nt = length(t);
		tout = [tout; t(2:nt)];
		xout = [xout;x(2:nt,:)];
		teout = [teout; te]; % Events at tstart are never reported.
		xeout = [xeout; xe];
		ieout = [ieout; ie];
	    	vV=zeros(length(x),1);
	    	vH=cos(x(:,1)).*x(:,4); 
	    	av_vel_perstep = [];
	    	mean_vel = mean(vH);
		output.speed = vH;
	    	av_vel_perstep = [av_vel_perstep mean_vel];
		
	    	% estimate of horizontal velocity of hips
		% Set the new initial conditions (after impact).
	    
		x0=transition_three_link(x(nt,:));

		% display some useful information to the user
		%disp(['step: ',num2str(i),', impact ratio:  ',num2str(x0(7)/x0(8))])

		% Only positions and velocities needed as initial conditions
		x0=x0(1:6);

		tstart = t(nt);
		if tstart>=tfinal
			break
	    end
	end
	%average_velocity = mean(av_vel_perstep);
	%output.speed = average_velocity;
end
