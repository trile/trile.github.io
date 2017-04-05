This is the development source for [trile.github.io](https://trile.github.io).
It is checked in as `source` brand in the same respository as the `master` brand is the generated version from Hexo.

Credit:
[How to setup a blog on github with Hexo](https://zirho.github.io/2016/06/04/hexo/)

To set up development, clone source as a single brand.

```commandline
git clone -b source --single-branch https://github.com/trile/trile.github.io.git trile.github.io.hexo
```

Create a new post

```commandline
$ hexo new "My new post"
```

Run dev server

```commandline
$ hexo server
```

Generate statis file

```commandline
$ hexo generate
```

Deploy to remote sites

```commandline
$ hexo deploy
```

Or generate and run in one command

```commandline
$ hexo generate -d
```

Add a theme

```
$ git submodule add {theme-github-url} themes/{theme-name}
```

Copy _config.yml.example to _config.yml

```commandline
$ cp themes/{theme-name}/_config.yml.example themes/{theme-name}/_config.yml
```
