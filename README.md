# Gyre

A server for AI image generation. Provides a backend service over multiple APIs
(GRPC and REST) for use by one or more frontends.

In particular, aims to be a local, open source implementation of the Stability AI APIs,
but also includes a lot of useful extensions.

Visit https://gyre.ai for features, installation instructions, API docs and more.

# Naming

```
Turning and turning in the widening gyre
The falcon cannot hear the falconer;
Things fall apart; the centre cannot hold;
Mere anarchy is loosed upon the world,
The blood-dimmed tide is loosed, and everywhere
The ceremony of innocence is drowned;
The best lack all conviction, while the worst
Are full of passionate intensity.
```

First stanza of The Second Coming by William Butler Yeats

AI is a force for disruption. But I am not so pesimistic as Yeats - the center will hold, even in the widening gyre.

# Thanks to / Credits:

- Seamless outpainting https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-beta2
- Additional schedulers https://github.com/hlky/diffusers
- K-Diffusion integration example https://github.com/Birch-san/diffusers/blob/1472b70194ae6d7e51646c0d6787815a5bc65f75/examples/community/play.py

# License

The main codebase is distributed under Apache-2.0. Dependancies are all compatible with that license, except as noted here:

- The nonfree directory contains code under some license that is more restrictive than Apache-2.0. Check the individual
  projects for license details. To fully comply with the Apache-2.0 license, remove this folder before release.
  + ToMe
  + Codeformer
- The Docker images contain a bunch of software under various open source licenses. The docker images tagged 'noncomm'
  include the nonfree folder, and so cannot be used commercially.

[![Stable Cabal Logo](stablecabal.png)](https://www.stablecabal.org/)
