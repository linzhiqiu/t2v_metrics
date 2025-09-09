# import open_clip


def get_model_collection_from_file(path):
    return [l.strip().split(",") for l in open(path).readlines()]


model_collection = {
}
