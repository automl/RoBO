function delay(Dt)

  % causes a system delay, useful in animation
  % delay() is called by 'full_simul.m'

  t0 = clock;
  while etime(clock,t0)<Dt,
  end;