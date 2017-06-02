package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.classifiers.functions.LinearRegression;

/**
 * Implements point-wise learner that can be used to implement logistic regression
 *
 */
public class PointwiseLearner extends Learner {

	@Override
	public Instances extractTrainFeatures(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {

		Instances dataset = null;

		/* Build attributes list */
		Instances X = null;

		/* Build X and Y matrices */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		X = new Instances("train_dataset", attributes, 0);
		/* Set last attribute as target */
		X.setClassIndex(X.numAttributes() - 1);
		int numAttributes = X.numAttributes();

		try {
			Map<Query,List<Document>> data_map = Util.loadTrainData (train_data_file);
			Map<String, Map<String, Double>> relData = Util.loadRelData(train_rel_file);

			Feature feature = new Feature(idfs);

			/* Add data */
			for (Query query : data_map.keySet()){        
				for (Document doc : data_map.get(query)){
					double[] features = feature.extractFeatureVector(doc, query);
					double[] instance = new double[numAttributes];
					for (int i = 0; i < features.length; ++i){
						instance[i] = features[i];
					}
					if (doc.url == null || relData.get(query.toString().trim()).get(doc.url) == null) {
						System.out.println("doc.url");
					}
					Double relScore = relData.get(query.toString().trim()).get(doc.url);
					instance[numAttributes - 1] = (relScore != null ? relScore : 0.0);
					Instance inst = new DenseInstance(1.0, instance); 
					X.add(inst);
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
		 */
		LinearRegression model = new LinearRegression();
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println(Arrays.toString(model.coefficients()));
		return model;
	}

	@Override
	public TestFeatures extractTestFeatures(String test_data_file,
			Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 * Create a TestFeatures object
		 * Build attributes list, instantiate an Instances object with the attributes
		 * Add data and populate the TestFeatures with the dataset and features
		 */
		Instances dataset = null;
		Map<Query, Map<Document, Integer>> index_map = new HashMap<Query, Map<Document, Integer>> ();
		int idx = 0;
		/* Build attributes list */
		Instances X = null;

		/* Build X and Y matrices */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		X = new Instances("test_dataset", attributes, 0);
		/* Set last attribute as target */
		X.setClassIndex(X.numAttributes() - 1);
		int numAttributes = X.numAttributes();

		try {
			Map<Query,List<Document>> data_map = Util.loadTrainData (test_data_file);
			Feature feature = new Feature(idfs);

			/* Add data */
			for (Query query : data_map.keySet()){
				index_map.put(query, new HashMap<Document, Integer> ());
				for (Document doc : data_map.get(query)){
					double[] features = feature.extractFeatureVector(doc, query);
					double[] instance = new double[numAttributes];
					for (int i = 0; i < features.length; ++i){
						instance[i] = features[i];
					}
					instance[numAttributes - 1] = 0.0;
					Instance inst = new DenseInstance(1.0, instance); 
					X.add(inst);
					index_map.get(query).put(doc, idx++);
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

		return new TestFeatures(dataset, index_map);
	}

	@Override
	public Map<Query, List<Document>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
		 */
		Instances dataset = tf.features;
		Map<Query, List<Document>> result = new HashMap<Query, List<Document>> ();
		Map<Query, Map<Document, Integer>> index_map = tf.index_map;
		try {
			
			for (Query query : index_map.keySet()) {
				int sz = index_map.get(query).size();
				List<Pair<Document,Double>> docAndScores = new ArrayList<Pair<Document,Double>> ();
				result.put(query, new ArrayList<Document> ());
				for (Document doc : index_map.get(query).keySet()) {
					double prediction = model.classifyInstance(dataset.instance(index_map.get(query).get(doc)));
					docAndScores.add(new Pair<Document, Double>(doc, prediction));
				}
				
				Collections.sort(docAndScores, new Comparator<Pair<Document,Double>>() {
			        @Override
			        public int compare(Pair<Document, Double> o1, Pair<Document, Double> o2) {
			        	return o1.getSecond() < o2.getSecond() ? 1 : o1.getSecond() == o2.getSecond() ? 0 : -1;
			        }
			      });
				for (Pair<Document, Double> das : docAndScores) {
					result.get(query).add(das.getFirst());
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

}
