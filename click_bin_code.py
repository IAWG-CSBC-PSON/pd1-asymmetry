@cli.command()
@click.argument("encodings", type=click.File("r"))
@click.argument("metadata", type=click.File("r"))
@click.option("--save_path",default=None,type=click.Path(file_okay=True, dir_okay=False, resolve_path=True))

def umap(encodings, metadata, save_path):
	run_umap(encodings, metadata, save_path)