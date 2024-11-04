def build_happy_dataset_for_train(dataset_name: str, tokenizer: Any) -> Dataset:
    train_dataset = Dataset(dataset_name)

    queries = []
    for file_name in glob.glob("/home/zt/data/open-images/train/processed_nn1/*.json") + glob.glob("/home/zt/data/open-images/train/processed_nn2/*.json"):
        with open(file_name) as f:
            queries.extend(json.load(f))
    index_img_ids = json.load(open(f"/home/zt/data/open-images/train/metadata/image_id.json"))
    index_image_folder = "/home/zt/data/open-images/train/data"

    # queries = []
    # file_path_pattern = "/home/zt/data/open-images/train/processed_nn1/response_results_batch_[0-9].json"       
    # files = glob.glob(file_path_pattern)
    # for file_name in files:
    #     with open(file_name) as f:
    #         queries.extend(json.load(f))
    # with open(f"/home/zt/data/open-images/train/metadata/image_id.json") as f:
    #     index_img_ids = json.load(f)[:200000]
    # index_image_folder = "/home/zt/data/open-images/train/data" 
    
    null_tokens = tokenize("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, index_img_id + ".jpg")
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['candidate']
        qtext = " and ".join(query['captions'])
        qimage_path = os.path.join(index_image_folder, query['candidate'] + ".jpg")
        ima = process_img(qimage_path, 224)
        qtokens = tokenize(qtext)
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                train_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Prepared index examples.")

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                train_dataset.query_examples.append(q_example)
                progress.update(1)
            
        print("Prepared query examples.")


    return train_dataset
