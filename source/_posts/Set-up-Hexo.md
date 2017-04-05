---
title: Set up Hexo
date: 2017-04-01 22:19:21
tags:
---

This is the dev source for my blog at [trile.github.io](http://google.com)
It is checked in as `source` brand in the same respository as the `master` brand is the generated version from Hexo.

**Credit:**

[How to setup a blog on github with Hexo](https://zirho.github.io/2016/06/04/hexo/)

To set up development, clone source as a single brand.

Create a new post

``` bash
$ hexo new "My new post"
```

``` bash
$ git clone -b source --single-branch https://github.com/trile/trile.github.io.git trile.github.io.hexo
```

``` javascript
var a = 3;
```

Add a theme

Copy _config.yml.example to _config.yml
