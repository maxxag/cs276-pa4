package cs276.pa4;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Skeleton code for the implementation of a BM25 Scorer in Task 2.
 */
public class BM25Scorer {

	String[] TFTYPES = {"url","title","body","header","anchor"};
	Map<String,Double> idfs;
	/*
	 *  TODO: You will want to tune these values
	 */
	double urlweight = 0.1;
	double titleweight  = 0.1;
	double bodyweight = 0.2;
	double headerweight = 0.4;
	double anchorweight = 0.2;
	Map <String, Double> weights = new HashMap<String, Double>();

	// BM25-specific weights
	double burl = 0.9;
	double btitle = 0.9;
	double bheader = 0.5;
	double bbody = 0.7;
	double banchor = 0.9;
	Map <String, Double> bweights = new HashMap<String, Double>();

	double k1 = 0.9;
	double pageRankLambda = 2.14;
	double pageRankLambdaPrime = 0.9;

	// query -> url -> document
	Map<Query,List<Document>> queryDict; 

	// BM25 data structures--feel free to modify these
	// Document -> field -> length
	Map<Document,Map<String,Double>> lengths;  

	// field name -> average length
	Map<String,Double> avgLengths;

	// Document -> pagerank score
	Map<Document,Double> pagerankScores; 

	/**
	 * Construct a BM25Scorer.
	 * @param idfs the map of idf scores
	 * @param queryDict a map of query to url to document
	 */
	public BM25Scorer(Map<String,Double> idfs, Map<Query,List<Document>> queryDict) {
		this.queryDict = queryDict;
		this.idfs =  idfs;
		weights.put("url", urlweight);
		weights.put("title", titleweight);
		weights.put("body", bodyweight);
		weights.put("header", headerweight);
		weights.put("anchor", anchorweight);

		bweights.put("url", burl);
		bweights.put("title", btitle);
		bweights.put("body", bbody);
		bweights.put("header", bheader);
		bweights.put("anchor", banchor);
		this.calcAverageLengths();
	}

	/**
	 * Set up average lengths for BM25, also handling PageRank.
	 */
	public void calcAverageLengths() {
		lengths = new HashMap<Document,Map<String,Double>>();
		avgLengths = new HashMap<String,Double>();
		pagerankScores = new HashMap<Document,Double>();

		/*
		 * TODO : Your code here
		 * Initialize any data structures needed, perform
		 * any preprocessing you would like to do on the fields,
		 * handle pagerank, accumulate lengths of fields in documents.        
		 */
		double totalLenUrl = 0;
		double totalLenTitle = 0;
		double totalLenBody = 0;
		double totalLenHeader = 0;
		double totalLenAnchor = 0;
		double totalDocCount = 0;

		/*
		 * TODO : Your code here
		 * Normalize lengths to get average lengths for
		 * each field (body, url, title, header, anchor)
		 */
		for (Query query : queryDict.keySet()) {
			for (Document d : queryDict.get(query)) {
				totalDocCount += 1;
				lengths.put(d, new HashMap<String, Double>());
				for (String tfType : this.TFTYPES) {
					if (tfType.equals("url")) {
						double urlLen = 0;
						if (d.url != null) {
							String[] urlTokens = d.url.split("[^\\p{Alnum}]");
							urlLen = urlTokens.length;
						}
						totalLenUrl += urlLen;
						lengths.get(d).put(tfType, (double)urlLen);
					} else if (tfType.equals("title")) {
						double titleLen = 0;
						if (d.title != null) {
							String[] titleTokens = d.title.split("\\s+");
							titleLen = titleTokens.length;
						}
						totalLenTitle += titleLen;
						lengths.get(d).put(tfType, (double)titleLen);
					} else if (tfType.equals("body")) {
						totalLenBody += d.body_length;
						lengths.get(d).put(tfType, (double)d.body_length);
					} else if (tfType.equals("header")) {
						double headersLen = 0;
						if (d.headers != null) {
							for (String header : d.headers) {
								String[] headerTokens = header.split("\\s+");
								headersLen += headerTokens.length;
							}
						}
						totalLenHeader += headersLen;
						lengths.get(d).put(tfType, headersLen);
					} else if (tfType.equals("anchor")) {
						double anchorsLen = 0;
						if (d.anchors != null) {
							for (String anchor : d.anchors.keySet()) {
								String[] anchorTokens = anchor.split("\\s+");
								anchorsLen += anchorTokens.length*d.anchors.get(anchor);
							}
						}
						totalLenAnchor += anchorsLen;
						lengths.get(d).put(tfType, anchorsLen);
					}
				}
				pagerankScores.put(d,(double)d.page_rank);
			}
		}
		for (String tfType : this.TFTYPES) {
			if (tfType.equals("url")) {
				avgLengths.put(tfType, totalLenUrl/totalDocCount);
			} else if (tfType.equals("title")) {
				avgLengths.put(tfType, totalLenTitle/totalDocCount);
			} else if (tfType.equals("body")) {
				avgLengths.put(tfType, totalLenBody/totalDocCount);
			} else if (tfType.equals("header")) {
				avgLengths.put(tfType, totalLenHeader/totalDocCount);
			} else if (tfType.equals("anchor")) {
				avgLengths.put(tfType, totalLenAnchor/totalDocCount);
			}
		}
	}

	/**
	 * Get the net score. 
	 * @param tfs the term frequencies
	 * @param q the Query 
	 * @param tfQuery
	 * @param d the Document
	 * @return the net score
	 */
	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q,Document d) {

		double score = 0.0;

		/*
		 * TODO : Your code here
		 * Use equation 5 in the writeup to compute the overall score
		 * of a document d for a query q.
		 */
		Set <String> queryTerms = new HashSet<String> ();
		queryTerms.addAll(q.queryWords);
		for (String term : queryTerms) {
			double wdt = 0;
			for (String tfType : TFTYPES) {
				wdt += weights.get(tfType) * tfs.get(tfType).get(term);
			}
			Double idf = this.idfs.get(term);
			if (idf == null) {
				idf = Math.log(98998.0);
			}
			score += wdt * idf / (k1 + wdt);
		}
		//score += pageRankLambda * Math.log(pageRankLambdaPrime + (pagerankScores.containsKey(d) ? pagerankScores.get(d) : 0));

		return score;
	}

	/**
	 * Do BM25 Normalization.
	 * @param tfs the term frequencies
	 * @param d the Document
	 * @param q the Query
	 */
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {
		/*
		 * TODO : Your code here
		 * Use equation 3 in the writeup to normalize the raw term frequencies
		 * in fields in document d.
		 */
		double len = 0, avglen = 1;
		for (String tfType : TFTYPES) {
			if (lengths.get(d).containsKey(tfType)) {
				len = lengths.get(d).get(tfType);
			}
			if (avgLengths.containsKey(tfType)) {
				avglen = avgLengths.get(tfType);

			}
			for (String word : tfs.get(tfType).keySet()) {
				double tf = 0;
				if (avglen == 0) {
					tf = 0;
				} else {
					double den = (double)1.0 + bweights.get(tfType) * (len/avglen - (double)1.0);
					tf = tfs.get(tfType).get(word)/den;
				}
				tfs.get(tfType).put(word, tf);
			}
		}
	}

	/**
	 * Write the tuned parameters of BM25 to file.
	 * Only used for grading purpose, you should NOT modify this method.
	 * @param filePath the output file path.
	 */
	private void writeParaValues(String filePath) {
		try {
			File file = new File(filePath);
			if (!file.exists()) {
				file.createNewFile();
			}
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			String[] names = {
					"urlweight", "titleweight", "bodyweight", 
					"headerweight", "anchorweight", "burl", "btitle", 
					"bheader", "bbody", "banchor", "k1", "pageRankLambda", "pageRankLambdaPrime"
			};
			double[] values = {
					this.urlweight, this.titleweight, this.bodyweight, 
					this.headerweight, this.anchorweight, this.burl, this.btitle, 
					this.bheader, this.bbody, this.banchor, this.k1, this.pageRankLambda, 
					this.pageRankLambdaPrime
			};
			BufferedWriter bw = new BufferedWriter(fw);
			for (int idx = 0; idx < names.length; ++ idx) {
				bw.write(names[idx] + " " + values[idx]);
				bw.newLine();
			}
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Get the similarity score.
	 * @param d the Document
	 * @param q the Query
	 * @return the similarity score
	 */
	public double getSimScore(Document d, Query q) {
		Map<String,Map<String, Double>> tfs = Util.getDocTermFreqs(d,q);
		this.normalizeTFs(tfs, d, q);

		// Write out the tuned BM25 parameters
		// This is only used for grading purposes.
		// You should NOT modify the writeParaValues method.
		//writeParaValues("bm25Para.txt");
		return getNetScore(tfs,q,d);
	}

}
