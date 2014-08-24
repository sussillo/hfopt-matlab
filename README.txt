hfopt-matlab
============

A parallel, cpu-based matlab implemention of the Hessian Free (HF) optimization
(feed forward networks, recurrent neural networks (RNN), multiplicative
recurrente neural networks (MRNN)).  Note that there is no CUDA interface under
the hood.  The main inspiration is that RNNs have trouble making use of GPU
parallelization anyways, because of the sequential dependence of one state on
the last.  This implementation simply uses data-parallel batches and then
averages afterwards.  In principle, if you had a matlab installation over many
machnies (and information transfer was OK), you could compute with extremely
large batches.  One further optimization that could be useful is to have batches
at the single run level.

In this repo you will find an optimizer that can train generic networks
according to function-based API.  I've implemented an RNN, an MRNN, and a
generic feed-forward network, though the last probably needs a little clean
up. The optimizer API provide a number of hooks to allow more general behavior,
such as allowing function hooks to define the data, and thuss allow arbitrary
data distributions that are changed in and out at will.  The optimizer also as a
function hook a plotting routine that is called after each HF iteration.  This
can be extremely useful for figuring out what is going on in your optimization.
For example, check out
examples/pathologicals/matlab/pathos_optional_plot_fun2.m.  

If you want to create your own network type, just start from one of the three
examples.  You have to do the normal stuff: define a network structure, the loss
function, the forward pass, the backward, and the dreaded Hessian-vector pass
(it's pretty awful).

* optimizer/ - routines to drive the optimizer.  The main function is hfopt2.m.
  The two comes from being the second redesign.

* examples/ - There are examples to be found in the examples/ subdirectory.
  Specifically, there are some of the classic pathological problems implemented.
  If you train them up, you can then use the drive_*.m scripts to analyze them,
  if that's your thing.  There is also a sinewave generator example.

* interfaces/ - A directory of really boring, tedious, and overly complicated
  interfaces.  This allows the optimizer to talk to the three different network
  types that are implemented.  If you wanted to build your own, you'd look at
  these and do some copy-and-paste.

* utils/ - some simple scripts to do simple things.

* dn/ - deep network - NOT WORKING RIGHT NOW.  TO BE FIXED SOON.  arbitrarily
  deep feed-forward networks. With modern GPUs, probably the only value of this
  code is doing research, since it's CPU based, matlab, and easy to visualize /
  manipulate.

* rnn/ - recurrent neural network - time is of the essence.  This is the main
  workhorse, and I've numerically verified the derivatives and Hv computations
  for non corner cases.  But a word of caution, the network structure net.xxx is
  awful.  I initially tried to use the feedforward network structure to build
  the RNN network, so it's laid out in three structrues.  The initial conditions
  are shoe-horned into the first layer (so you'll see a warning about a bug.)
  The upshot is that I'm pretty happy with this code but the RNN net structure
  itself could stand to be rewritten.  Also, I can't vouch that all the
  analyeses scripts work, but the best place to start is
  drive_fp_analysis_pathos_add.m

* mrnn/ - multiplicative recurrent neural network


* Additional - I'm fond of working with Matlab cells.  If you don't know what
  are, then you might be annoyed at the layout of the script files.  For
  example, there is a call to matlabpool at the top of most top-level script
  functions.  If you run these from the command line, after the first script
  call, it'll error out.  But I use cells, so I go from one cell to the next.
  Just sayin'.
