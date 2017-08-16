This is the development source for [trile.github.io](https://trile.github.io).
It is checked in as `source` brand in the same respository as the `master` brand is the generated version from Hexo.

Credit:
[How to setup a blog on github with Hexo](https://zirho.github.io/2016/06/04/hexo)

**Notes:** If you want to follow that article to set up a new blog, remember to exclude `.deploy_git` and `db.json` using `.gitignore`, otherwise they will mess up hexo git deploy.

To set up development, clone source as a single brand.

```bash
> git clone -b source https://github.com/trile/trile.github.io.git trile.github.io.hexo
```

Create a new post

```bash
> hexo new "My new post"
```

Run dev server

```bash
> hexo server
```

Generate statis file

```bash
> hexo generate
```

Deploy to remote sites

```bash
> hexo deploy
```

Or generate and run in one command

```bash
> hexo generate -d
```

Add a theme

```bash
> git submodule add {theme-github-url} themes/{theme-name}
```

Copy _config.yml.example to _config.yml

```bash
> cp themes/{theme-name}/_config.yml.example themes/{theme-name}/_config.yml
```

More information can be obtain at [Hexo documentation](https://hexo.io/docs/)
