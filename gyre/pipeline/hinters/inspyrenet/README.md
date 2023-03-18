From https://github.com/plemeri/transparent-background/tree/main/transparent_background

Used under MIT license

Modified to:
- Update import reference
- Change modules/layers.py/ImagePyramid and modules/layers.py/Transition to be
  nn.Modules with their kernels registered as buffers so manager.py can move them.
  Previously these were just bare classes with to()/cuda() overrides and so would not 
  move when clone_model was used.
