# equate.featurize.text

Dense text-embedding featurizers, behind optional extras.

Each is a *lazily-built* strategy: the heavy library is imported only when the
strategy is actually constructed, via `equate._dependencies.require()`, so a
missing extra raises a friendly `MissingDependencyError` instead of breaking
`import equate`. A representative set is provided here; more embedders (bge-m3, e5,
nomic, cohere, voyage, and image/audio modalities) are added the same way as their
stages land — see the `equate-dev-add-strategy` skill.

### Functions

| [`sbert_featurizer`](#equate.featurize.text.sbert_featurizer)(\*[, model])   | Sentence-Transformers (SBERT) embedder — requires `equate[embeddings]`.   |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`openai_featurizer`](#equate.featurize.text.openai_featurizer)(\*[, model])  | OpenAI embeddings — requires `equate[api]` and an `OPENAI_API_KEY`.       |

### equate.featurize.text.openai_featurizer(, model='text-embedding-3-small')

OpenAI embeddings — requires `equate[api]` and an `OPENAI_API_KEY`.

Returns a batch featurizer `texts -> embedding matrix`.

### equate.featurize.text.sbert_featurizer(, model='sentence-transformers/all-MiniLM-L6-v2')

Sentence-Transformers (SBERT) embedder — requires `equate[embeddings]`.

Returns a batch featurizer `texts -> L2-normalized embedding matrix`.
