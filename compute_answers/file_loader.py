# -*- coding: utf-8 -*-


def generate_dataset(dataset, test = False):
    """
    Takes as an input a HuggingFace Dataset with the format specified in the training
    file and transform it so that each title has associated a single question, context
    and a single answer as well.
    this implies a replication of the titles (since one title has more questions) and
    questions/context (since one question/context can hold more than one answer)

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset.
    test : boolean, optional
        True if the answers of the questions are not present. The default is False.

    Yields
    ------
    id_ : str
        Question's' id.
    dict
        Other dataset fields: title, context, question, id.
        If test is false then also the answers with its relative fields
        (answer_start, text).

    """
    for data in dataset["train"]:
        title = data.get("title", "").strip()
        for paragraph in data["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                # Handling questions
                question = qa["question"].strip()
                id_ = qa["id"]
                # Answers won't be present in the testing 
                if not test:
                    # handling answers
                    for answer in qa["answers"]:
                        answer_start = [answer["answer_start"]]
                    for answer in qa["answers"]:
                        answer_text = [answer["text"].strip()]
                    
                    yield id_, {
                        "title": title,
                        "context": context,
                        "question": question,
                        "id": id_,
                        "answers": {
                            "answer_start": answer_start,
                            "text": answer_text,
                        },
                    }
                else:
                    yield id_, {
                        "title": title,
                        "context": context,
                        "question": question,
                        "id": id_,
                    }