package cs276.pa4;
import weka.core.Instances;
import java.util.Map;

/** 
 *  A sample class to store the result
 */
public class TestFeatures {
	/* Test features */
	Instances features; 

	/* Associate query-doc pair to its index within FEATURES instances
	 * {query -> {doc -> index}}
	 * 
	 * For example, you can get the feature for a pair of (query, doc) using:
	 *   features.get(index_map.get(query).get(doc));
	 * */
	Map<Query, Map<Document, Integer>> index_map;
	Map<Query, Map<Pair<Document, Document>, Integer>> svm_index_map;
	public TestFeatures (Instances fetaures, Map<Query, Map<Document, Integer>> index_map) {
		this.features = fetaures;
		this.index_map = index_map;
	}
	public TestFeatures (Instances fetaures, Map<Query, Map<Pair<Document,Document>, Integer>> svm_map, boolean isSVM) {
		this.features = fetaures;
		this.svm_index_map = svm_map;
	}
	public TestFeatures() {
		super();
	}
}
