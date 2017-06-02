package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Implements Pairwise learner that can be used to train SVM
 *
 */
public class PairwiseLearner extends Learner {
	private LibSVM model;
	public boolean includeBM25 = false;
	public boolean includeSmallestWindow = false;
	public boolean includePageRank = false;
	int bm25idx = -1;
	int pagerankidx = -1;
	int swidx = -1;
	public PairwiseLearner(boolean isLinearKernel){
		try{
			model = new LibSVM();
		} catch (Exception e){
			e.printStackTrace();
		}

		if(isLinearKernel){
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
	}
	public PairwiseLearner(boolean isLinearKernel, boolean includeBM25, boolean includeSmallestWindow, boolean includePageRank){
		try{
			model = new LibSVM();
		} catch (Exception e){
			e.printStackTrace();
		}
		
		if(isLinearKernel){
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
		this.includeBM25 = includeBM25;
		this.includeSmallestWindow = includeSmallestWindow;
		this.includePageRank = includePageRank;
	}
	public PairwiseLearner(double C, double gamma, boolean isLinearKernel, boolean includeBM25, boolean includeSmallestWindow, boolean includePageRank){
		try{
			model = new LibSVM();
		} catch (Exception e){
			e.printStackTrace();
		}
		model.setCost(C);
		model.setGamma(gamma);
		if(isLinearKernel){
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
		this.includeBM25 = includeBM25;
		this.includeSmallestWindow = includeSmallestWindow;
		this.includePageRank = includePageRank;
	}

	public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
		try{
			model = new LibSVM();
		} catch (Exception e){
			e.printStackTrace();
		}

		model.setCost(C);
		model.setGamma(gamma); // only matter for RBF kernel
		if(isLinearKernel){
			model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		}
	}

	@Override
	public Instances extractTrainFeatures(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here:
		 * Get signal file 
		 * Construct output dataset of type Instances
		 * Add new attribute  to store relevance in the train dataset
		 * Populate data
		 */
		Instances dataset = null;

		/* Build attributes list */
		Instances X = null;
		int featurectr = 0;

		/* Build X and Y matrices */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		if (this.includeBM25) {
			attributes.add(new Attribute("bm25_w"));
			bm25idx = 5 + featurectr;
			featurectr++;
		}
		if (this.includeSmallestWindow) {
			attributes.add(new Attribute("sw_w"));
			swidx = 5 + featurectr;
			featurectr++;
		}
		if (this.includePageRank) {
			attributes.add(new Attribute("pagerank_w"));
			pagerankidx = 5 + featurectr;
			featurectr++;
		}
		ArrayList<String> labels = new ArrayList<String>();
		labels.add("+1");
		labels.add("-1");
		attributes.add(new Attribute("relevance", labels));
		int pos_inst = 0, neg_inst = 0;
		X = new Instances("train_dataset", attributes, 0);
		/* Set last attribute as target */
		X.setClassIndex(X.numAttributes() - 1);
		int numAttributes = X.numAttributes();

		try {
			Map<Query,List<Document>> data_map = Util.loadTrainData (train_data_file);
			Map<String, Map<String, Double>> relData = Util.loadRelData(train_rel_file);

			Feature feature = new Feature(idfs);
			BM25Scorer bm25 = null;
			SmallestWindowScorer swscorer = null;
			if (this.includeBM25) {
				bm25 = new BM25Scorer(idfs, data_map);
			}
			if (this.includeSmallestWindow) {
				swscorer = new SmallestWindowScorer(idfs, data_map);
			}

			/* Add data */
			for (Query query : data_map.keySet()){    
				for (int i = 0; i < data_map.get(query).size(); i++) {
					Document doc1 = data_map.get(query).get(i);
					double rel1 = relData.get(query.toString().trim()).get(doc1.url);
					double[] features1 = feature.extractFeatureVector(doc1, query);
					double bm25score1 = 0, pagerank1 = 0, swscore1 = 0;
					if (this.includeBM25) {
						bm25score1 = bm25.getSimScore(doc1, query);
					}
					if (this.includeSmallestWindow) {
						swscore1 = swscorer.getSimScore(doc1, query);
						//System.out.println(swscore1);
					}
					if (this.includePageRank) {
						pagerank1 = doc1.page_rank;
					}
					for (int j = i+1; j < data_map.get(query).size(); j++) {
						Document doc2 = data_map.get(query).get(j);
						double rel2 = relData.get(query.toString().trim()).get(doc2.url);
						double[] features2 = feature.extractFeatureVector(doc2, query);
						double bm25score2 = 0, pagerank2 = 0, swscore2 = 0;
						if (this.includeBM25) {
							bm25score2 = bm25.getSimScore(doc2, query);
						}
						if (this.includeSmallestWindow) {
							swscore2 = swscorer.getSimScore(doc2, query);
						}
						if (this.includePageRank) {
							pagerank2 = doc2.page_rank;
						}
						double[] instance = new double[numAttributes];
						if (rel1 > rel2) {
							if (pos_inst > neg_inst) {
								for (int k = 0; k < features1.length; ++k) {
									instance[k] = features2[k] - features1[k];   //negative
								}
								if (this.includeBM25) {
									instance[bm25idx] = bm25score2 - bm25score1;
								}
								if (this.includeSmallestWindow) {
									instance[swidx] = swscore2 - swscore1;
								}
								if (this.includePageRank) {
									instance[pagerankidx] = pagerank2 - pagerank1;
								}
								instance[numAttributes - 1] = 
										X.attribute(numAttributes-1).indexOfValue("-1");
								neg_inst++;
							} else {
								for (int k = 0; k < features1.length; ++k) {
									instance[k] = features1[k] - features2[k];   //positive
								}
								if (this.includeBM25) {
									instance[bm25idx] = bm25score1 - bm25score2;
								}
								if (this.includeSmallestWindow) {
									instance[swidx] = swscore1 - swscore2;
								}
								if (this.includePageRank) {
									instance[pagerankidx] = pagerank1 - pagerank2;
								}
								instance[numAttributes - 1] = 
										X.attribute(numAttributes-1).indexOfValue("+1");
								pos_inst++;
							}
							Instance inst = new DenseInstance(1.0, instance); 
							X.add(inst);
						} else if (rel2 > rel1) {
							if (pos_inst > neg_inst) {
								for (int k = 0; k < features1.length; ++k) {
									instance[k] = features1[k] - features2[k];   //negative
								}
								if (this.includeBM25) {
									instance[bm25idx] = bm25score1 - bm25score2;
								}
								if (this.includeSmallestWindow) {
									instance[swidx] = swscore1 - swscore2;
								}
								if (this.includePageRank) {
									instance[pagerankidx] = pagerank1 - pagerank2;
								}
								instance[numAttributes - 1] = 
										X.attribute(numAttributes-1).indexOfValue("-1");
								neg_inst++;
							} else {
								for (int k = 0; k < features1.length; ++k) {
									instance[k] = features2[k] - features1[k];   //positive
								}
								if (this.includeBM25) {
									instance[bm25idx] = bm25score2 - bm25score1;
								}
								if (this.includeSmallestWindow) {
									instance[swidx] = swscore2 - swscore1;
								}
								if (this.includePageRank) {
									instance[pagerankidx] = pagerank2 - pagerank1;
								}
								instance[numAttributes - 1] = 
										X.attribute(numAttributes-1).indexOfValue("+1");
								pos_inst++;
							}
							Instance inst = new DenseInstance(1.0, instance); 
							X.add(inst);
						}
						
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		/* Conduct standardization on X */
		Standardize filter = new Standardize();
		//    Normalize filter = new Normalize(); filter.setScale(2.0); filter.setTranslation(-1.0); // scale values to [-1, 1]
		try {
			filter.setInputFormat(X); 
			dataset = Filter.useFilter(X, filter);
		} catch (Exception e) {
			e.printStackTrace();
		} 

		return dataset;
	}

	@Override
	public Classifier training(Instances dataset) {
		/*
		 * @TODO: Your code here
		 * Build classifer
		 */
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			e.printStackTrace();
		}
		//System.out.println(Arrays.toString(model.coefficients()));
		return model;
	}

	@Override
	public TestFeatures extractTestFeatures(String test_data_file,
			Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 * Use this to build the test features that will be used for testing
		 */
		Instances dataset = null;
		Map<Query, Map<Pair<Document, Document>, Integer>> svm_index_map = new HashMap<Query, Map<Pair<Document, Document>, Integer>> ();
		int idx = 0;
		/* Build attributes list */
		Instances X = null;
		int featurectr = 0;
		/* Build X and Y matrices */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		if (this.includeBM25) {
			attributes.add(new Attribute("bm25_w"));
			bm25idx = 5 + featurectr;
			featurectr++;
		}
		if (this.includeSmallestWindow) {
			attributes.add(new Attribute("sw_w"));
			swidx = 5 + featurectr;
			featurectr++;
		}
		if (this.includePageRank) {
			attributes.add(new Attribute("pagerank_w"));
			pagerankidx = 5 + featurectr;
			featurectr++;
		}
		ArrayList<String> labels = new ArrayList<String>();
		labels.add("+1");
		labels.add("-1");
		attributes.add(new Attribute("relevance", labels));
		X = new Instances("test_dataset", attributes, 0);
		/* Set last attribute as target */
		X.setClassIndex(X.numAttributes() - 1);
		int numAttributes = X.numAttributes();

		try {
			Map<Query,List<Document>> data_map = Util.loadTrainData (test_data_file);
			Feature feature = new Feature(idfs);
			BM25Scorer bm25 = null;
			SmallestWindowScorer swscorer = null;
			if (this.includeBM25) {
				bm25 = new BM25Scorer(idfs, data_map);
			}
			if (this.includeSmallestWindow) {
				swscorer = new SmallestWindowScorer(idfs, data_map);
			}
			/* Add data */
			for (Query query : data_map.keySet()){
				svm_index_map.put(query, new HashMap<Pair<Document, Document>, Integer> ());
				for (int i = 0; i < data_map.get(query).size(); i++) {
					Document doc1 = data_map.get(query).get(i);
					double[] features1 = feature.extractFeatureVector(doc1, query);
					double bm25score1 = 0, pagerank1 = 0, swscore1 = 0;
					if (this.includeBM25) {
						bm25score1 = bm25.getSimScore(doc1, query);
					}
					if (this.includeSmallestWindow) {
						swscore1 = swscorer.getSimScore(doc1, query);
					}
					if (this.includePageRank) {
						pagerank1 = doc1.page_rank;
					}
					for (int j = i+1; j < data_map.get(query).size(); j++) {
						Document doc2 = data_map.get(query).get(j);
						double[] features2 = feature.extractFeatureVector(doc2, query);
						double bm25score2 = 0, pagerank2 = 0, swscore2 = 0;
						if (this.includeBM25) {
							bm25score2 = bm25.getSimScore(doc2, query);
						}
						if (this.includeSmallestWindow) {
							swscore2 = swscorer.getSimScore(doc2, query);
						}
						if (this.includePageRank) {
							pagerank2 = doc2.page_rank;
						}
						double[] instance = new double[numAttributes];
						for (int k = 0; k < features1.length; ++k){
							instance[k] = features1[k] - features2[k];
						}
						if (this.includeBM25) {
							instance[bm25idx] = bm25score1 - bm25score2;
						}
						if (this.includeSmallestWindow) {
							instance[swidx] = swscore1 - swscore2;
						}
						if (this.includePageRank) {
							instance[pagerankidx] = pagerank1 - pagerank2;
						}
						instance[numAttributes - 1] = 0.0;
						Instance inst = new DenseInstance(1.0, instance); 
						X.add(inst);
						svm_index_map.get(query).put(new Pair<Document, Document>(doc1, doc2), idx);
						idx++;
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		/* Conduct standardization on X */
		Standardize filter = new Standardize();
		//    Normalize filter = new Normalize(); filter.setScale(2.0); filter.setTranslation(-1.0); // scale values to [-1, 1]
		try {
			filter.setInputFormat(X); 
			dataset = Filter.useFilter(X, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return new TestFeatures(dataset, svm_index_map, true);
	}

	@Override
	public Map<Query, List<Document>> testing(TestFeatures tf,
			Classifier model) {
		Instances dataset = tf.features;
		int numAttributes = dataset.numAttributes();
		//System.out.println(dataset.attribute(numAttributes - 1));
		Map<Query, List<Document>> result = new HashMap<Query, List<Document>> ();
		Map<Query, Map<Pair<Document, Document>, Integer>> svm_index_map = tf.svm_index_map;
		try {

			for (Query query : svm_index_map.keySet()) {
				int sz = svm_index_map.get(query).size();
				Set<Document> docsSet = new HashSet<Document> ();
				ArrayList<Document> docs = new ArrayList<Document>();
				for (Pair<Document, Document> docPair : svm_index_map.get(query).keySet()) {
					docsSet.add(docPair.getFirst());
					docsSet.add(docPair.getSecond());
				}
					
				//result.put(query, new ArrayList<Document> ());
				docs.addAll(docsSet);				
				Collections.sort(docs, new Comparator<Document>() {
					@Override
					public int compare(Document d1, Document d2) {
						int neg = 1; // neg inactive
						Integer idx = null;
						Pair<Document, Document> p = null;
						if ((idx = svm_index_map.get(query).get(p = new Pair<Document, Document> (d1, d2))) == null) {
							idx = svm_index_map.get(query).get(p = new Pair<Document, Document> (d2, d1));
							if (idx == null) {
								System.out.println("idx null");
							}
							neg= -1; // neg active
							//System.out.println("neg -1");
						}
						double prediction = Double.MIN_VALUE;
						String predictStr = "";
						try {
							prediction = model.classifyInstance(dataset.instance(idx));
							//System.out.println((int) prediction);
							predictStr = dataset.attribute(numAttributes-1).value((int) prediction);
							//System.out.println("predictStr " + dataset.attribute(numAttributes-1).value((int) prediction) + "maxx");
							//System.out.println(prediction);
						} catch (Exception e) {
							e.printStackTrace();
						}

						if (predictStr.equals("+1")) return -1 * neg;
						else if (predictStr.equals("-1")) return 1 * neg;
						else {
							System.out.println("returning default value of prediction");
							return 0;
						}
					}
				});
				result.put(query, docs);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

}
