package cs276.pa4;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;


/**
 * A skeleton for implementing the Smallest Window scorer in Task 3.
 * Note: The class provided in the skeleton code extends BM25Scorer in Task 2. However, you don't necessarily
 * have to use Task 2. (You could also use Task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead.)
 * Also, feel free to modify or add helpers inside this class.
 */
public class SmallestWindowScorer extends BM25Scorer {
	//public class SmallestWindowScorer extends CosineSimilarityScorer {

	public double B = 0.236;

	public SmallestWindowScorer(Map<String, Double> idfs, Map<Query,List <Document>> queryDict) {
		//public SmallestWindowScorer(Map<String, Double> idfs) {
		super(idfs, queryDict);
		//super(idfs);
	}

	private int getSmallestWindow(List<Pair<Integer, String>> wpa, Set<String> querySet) {
		ArrayList<Pair<Integer, String>> currBlock = new ArrayList<Pair<Integer, String>> ();
		boolean propertyViolated = false;
		int smallestSoFar = Integer.MAX_VALUE;
		for (int iter = 0; iter < wpa.size(); iter++) {
			currBlock.add(wpa.get(iter));
			//check property
			if (inList(currBlock.get(0).getSecond(), currBlock.subList(1, currBlock.size()))) {
				propertyViolated = true;
				//if property violated, normalize
				while(!currBlock.isEmpty() && propertyViolated) {
					currBlock.remove(0);
					if (!currBlock.isEmpty() && inList(currBlock.get(0).getSecond(), currBlock.subList(1, currBlock.size()))) {
						propertyViolated = true;
					} else {
						propertyViolated = false;
					}
				}
			}
			//if all words record size
			Set<String> currBlockUniqueElems = new HashSet<String>();
			for (int blockIter = 0; blockIter < currBlock.size(); blockIter++) {
				currBlockUniqueElems.add(currBlock.get(blockIter).getSecond());
			}
			if (!currBlock.isEmpty() && currBlockUniqueElems.size() == querySet.size()) {
				if (currBlock.get(currBlock.size()-1).getFirst() - currBlock.get(0).getFirst() + 1 < smallestSoFar) {
					smallestSoFar = currBlock.get(currBlock.size()-1).getFirst() - currBlock.get(0).getFirst() + 1;
				}
			}		  
		}
		if (smallestSoFar < Integer.MAX_VALUE) {
			return smallestSoFar;
		}
		return -1;

	}

	private int getWindowField(Document d, Set<String> querySet) {
		Map<String, Integer> smallestWindows = new HashMap<String,Integer>();
		int smallestSoFar = Integer.MAX_VALUE;
		if (d.url != null) {
			ArrayList<Pair<Integer, String>> wpa = new ArrayList<Pair<Integer, String>>();
			String[] urlTokens = d.url.split("[^\\p{Alnum}]");
			for (int i = 0; i < urlTokens.length; i++) {
				if (querySet.contains(urlTokens[i].toLowerCase())) {
					wpa.add(new Pair<Integer, String>(i, urlTokens[i].toLowerCase()));
				}
			}
			smallestWindows.put("url", getSmallestWindow(wpa, querySet));
		}
		if (d.title != null) {
			ArrayList<Pair<Integer, String>> wpa = new ArrayList<Pair<Integer, String>>();
			String[] titleTokens = d.title.split("\\s+");
			for (int i = 0; i < titleTokens.length; i++) {
				if (querySet.contains(titleTokens[i].toLowerCase())) {
					wpa.add(new Pair<Integer, String>(i, titleTokens[i].toLowerCase()));
				}
			}
			smallestWindows.put("title", getSmallestWindow(wpa, querySet));
		}
		if (d.headers != null) {
			String combinedHeader = "";
			for (String header : d.headers) {
				combinedHeader += header + " ";
			}
			combinedHeader = combinedHeader.trim();
			ArrayList<Pair<Integer, String>> wpa = new ArrayList<Pair<Integer, String>>();
			String[] headerTokens = combinedHeader.split("\\s+");
			for (int i = 0; i < headerTokens.length; i++) {
				if (querySet.contains(headerTokens[i].toLowerCase())) {
					wpa.add(new Pair<Integer, String>(i, headerTokens[i].toLowerCase()));
				}
			}
			smallestWindows.put("header", getSmallestWindow(wpa, querySet));
		}
		if (d.anchors != null) {
			int smallestAnchorWindow = Integer.MAX_VALUE;
			int windowSize = Integer.MAX_VALUE;
			for (String anchor : d.anchors.keySet()) {
				ArrayList<Pair<Integer, String>> wpa = new ArrayList<Pair<Integer, String>>();
				String[] anchorTokens = anchor.split("\\s+");
				for (int i = 0; i < anchorTokens.length; i++) {
					if (querySet.contains(anchorTokens[i].toLowerCase())) {
						wpa.add(new Pair<Integer, String>(i, anchorTokens[i].toLowerCase()));
					}
				}
				windowSize = getSmallestWindow(wpa, querySet);
				if (windowSize != -1 && windowSize < smallestAnchorWindow) {
					smallestAnchorWindow = windowSize;
				}
			}
			smallestWindows.put("anchor", (smallestAnchorWindow < Integer.MAX_VALUE) ? smallestAnchorWindow : -1);
		}

		for(String field : smallestWindows.keySet()) {
			if (smallestWindows.get(field) != -1 && smallestWindows.get(field) < smallestSoFar) {
				smallestSoFar = smallestWindows.get(field);
			}
		}
		if (smallestSoFar < Integer.MAX_VALUE) {
			return smallestSoFar;
		}
		return -1;
	}

	private int getWindowBody(Document d, Set<String> querySet) {
		for (String word : querySet) {
			if (d.body_hits == null || !d.body_hits.keySet().contains(word)) {
				return -1;
			}
		}
		LinkedList<ArrayList<Pair<Integer, String>>> wordPosArrQueue = new LinkedList<ArrayList<Pair<Integer, String>>>();
		//create a large sorted array of Pair<position,word>
		for (String word : d.body_hits.keySet()) {
			ArrayList<Pair<Integer, String>> wordPosArr = new ArrayList<Pair<Integer, String>> ();
			for (int pos : d.body_hits.get(word)) {
				wordPosArr.add(new Pair<Integer,String> (pos,word.toLowerCase()));
			}
			wordPosArrQueue.add(wordPosArr);
		}

		while (true) {
			if (wordPosArrQueue.size() <= 1)
				break;

			ArrayList<Pair<Integer, String>> wpa1 = wordPosArrQueue.removeFirst();
			ArrayList<Pair<Integer, String>> wpa2 = wordPosArrQueue.removeFirst();
			ArrayList<Pair<Integer, String>> combwpa = new ArrayList<Pair<Integer, String>>();
			int i = 0, j = 0;
			while (i < wpa1.size() && j < wpa2.size()) {
				Integer pos1 = wpa1.get(i).getFirst();
				Integer pos2 = wpa2.get(j).getFirst();

				if (pos1 <= pos2) {
					combwpa.add(wpa1.get(i));
					i++;
				} else {
					combwpa.add(wpa2.get(j));
					j++;
				}
			}
			if (i >= wpa1.size()) {
				while (j < wpa2.size()) {
					combwpa.add(wpa2.get(j));
					j++;
				}
			} else if (j >= wpa2.size()) {
				while (i < wpa1.size()) {
					combwpa.add(wpa1.get(i));
					i++;
				}
			}
			wordPosArrQueue.add(combwpa);
		}
		if (!wordPosArrQueue.isEmpty()) {
			return getSmallestWindow(wordPosArrQueue.removeFirst(), querySet);
		}
		return -1;	  
	}

	boolean inList (String str, List<Pair<Integer, String>> wpa) {
		if(wpa.isEmpty())
			return false;
		for (int i = 0; i < wpa.size(); i++) {
			if (wpa.get(i).getSecond().equals(str)) {
				return true;
			}
		}
		return false;
	}

	/**
	 * get smallest window of one document and query pair.
	 * @param d: document
	 * @param q: query
	 */  
	private int getWindow(Document d, Set<String> querySet) {
		/*
		 * @//TODO : Your code here
		 */
		int smallestWindowSize = Integer.MAX_VALUE;
		int size = getWindowBody(d, querySet);
		if (size != -1 && size < smallestWindowSize) {
			smallestWindowSize = size;
		}
		size = getWindowField(d, querySet);
		if (size != -1 && size < smallestWindowSize) {
			smallestWindowSize = size;
		}
		if (smallestWindowSize < Integer.MAX_VALUE) {
			return smallestWindowSize;
		}
		return -1;
	}


	/**
	 * get boost score of one document and query pair.
	 * @param d: document
	 * @param q: query
	 */  
	private double getBoostScore (Document d, Query q) {
		Set<String> querySet = new HashSet<String>();
		for (String word : q.queryWords) {
			querySet.add(word.toLowerCase());
		}
		int smallestWindow = getWindow(d, querySet);
		if (smallestWindow == -1 || smallestWindow == Integer.MAX_VALUE) {
			return 1.0;
		}
		double boostScore = 0;
		boostScore = B * (double)1.0/(smallestWindow + 1 - querySet.size());
		boostScore += 1.0;
		/*
		 * @//TODO : Your code here, calculate the boost score.
		 *
		 */
		return boostScore;
	}

	@Override
	public double getSimScore(Document d, Query q) {
		Map<String,Map<String, Double>> tfs = Util.getDocTermFreqs(d,q);
		this.normalizeTFs(tfs, d, q);
		double boost = getBoostScore(d, q);
		double rawScore = this.getNetScore(tfs, q, d);
		return boost * rawScore;
	}

}
