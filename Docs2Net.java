package GoogleSnippets;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;

/**
 * Transalte docs to net
 * 
 * @author wx
 *
 */
public class Docs2Net {
	private static final int TOP_K_NEIGHBOR = 200;
	public HashMap<String, ArrayList<Integer>> mapWordDoclist = new HashMap<String, ArrayList<Integer>>(100000);
	public HashMap<Integer, String> mapIdContent = new HashMap<Integer, String>(20000);

	public Docs2Net(String path) {
		this.loadWordDocR(path);
	}

	public void loadWordDocR(String path) {
		try {
			BufferedReader bd = new BufferedReader(new FileReader(new File(path)));
			String line = null;
			int docNum = 0;
			while ((line = bd.readLine()) != null) {
				// 以空格分隔改行，并将单词添加入set结构中
				docNum++;
				this.mapIdContent.put(docNum, line);

				String[] words = line.split(" ");
				for (int i = 1; i < words.length; i++) {
					String word = words[i].split(":")[0].trim();
					if (this.mapWordDoclist.containsKey(word)) {
						ArrayList<Integer> docs = this.mapWordDoclist.get(word);
						docs.add(docNum);
						this.mapWordDoclist.put(word, docs);
					} else {
						ArrayList<Integer> docs = new ArrayList<Integer>();
						docs.add(docNum);
						this.mapWordDoclist.put(word, docs);
					}
				}
			}

			bd.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void docs2Net(String path) {
		try {
			BufferedReader bd = new BufferedReader(new FileReader(new File(path)));
			BufferedWriter bw = new BufferedWriter(new FileWriter(new File(path + ".heatkernel.net"), true));

			String line = null;
			int docNum = 0;
			while ((line = bd.readLine()) != null) {
				// 以空格分隔改行，并将单词添加入set结构中
				docNum++;
				String[] words = line.split(" ");

				List<Integer> listEdgeNode = new ArrayList<Integer>(1000);

				for (int i = 1; i < words.length; i++) {
					String word = words[i].split(":")[0].trim();

					if (this.mapWordDoclist.containsKey(word)) {
						ArrayList<Integer> docs = this.mapWordDoclist.get(word);
						for (int docid : docs) {
							if (!listEdgeNode.contains(docid)) {
								listEdgeNode.add(docid);
							}
						}
					}
				}

				Collections.sort(listEdgeNode);
				for (int docid : listEdgeNode) {
//					double sim = this.computeSimilarity(docNum, docid);
					double sim = this.computeSimilarityHeatKernel(docNum, docid);
					if (sim >= 0.001) {
						bw.write(docNum + " " + docid + " " + sim);
						bw.newLine();
					}
				}

			}

			bw.flush();
			bd.close();
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * 仅仅保留top-k个邻居节点的数据
	 * 
	 * @param path
	 */
	public void docs2NetTopk(String path) {
		try {
			BufferedReader bd = new BufferedReader(new FileReader(new File(path)));
			BufferedWriter bw = new BufferedWriter(
					new FileWriter(new File(path + ".top" + this.TOP_K_NEIGHBOR + ".net"), true));

			String line = null;
			int docNum = 0;
			while ((line = bd.readLine()) != null) {
				// 以空格分隔改行，并将单词添加入set结构中
				docNum++;
				if(docNum%1000==0){
					System.out.println("We have finished "+docNum +" docs!");
				}
				String[] words = line.split(" ");

				List<Integer> listEdgeNode = new ArrayList<Integer>(1000);

				for (int i = 1; i < words.length; i++) {
					String word = words[i].split(":")[0].trim();

					if (this.mapWordDoclist.containsKey(word)) {
						ArrayList<Integer> docs = this.mapWordDoclist.get(word);
						for (int docid : docs) {
							if (!listEdgeNode.contains(docid)) {
								listEdgeNode.add(docid);
							}
						}
					}
				}

				Collections.sort(listEdgeNode);
				Map<Integer, Double> map = new TreeMap<Integer, Double>();
				for (int docid : listEdgeNode) {
					double sim = this.computeSimilarity(docNum, docid);
					map.put(docid, sim);
					// //unweighted graph
					// bw.write(docNum + " " + docid);
					// bw.newLine();
				}

				// 降序排序
				List<Map.Entry<Integer, Double>> list = new ArrayList<Map.Entry<Integer, Double>>(map.entrySet());
				Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
					public int compare(Entry<Integer, Double> o1, Entry<Integer, Double> o2) {
						return o2.getValue().compareTo(o1.getValue());
					}
				});

				// 将相似度的top-k个邻居节点数据写入到文件
				int k = 1;
				for (Map.Entry<Integer, Double> mapping : list) {

					bw.write(docNum + " " +mapping.getKey() + " " + mapping.getValue());
					bw.newLine();
					k++;
					if (k > this.TOP_K_NEIGHBOR) {
						break;
					}
				}

			}

			bw.flush();
			bd.close();
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * 计算doc1和doc2的语义相似度，采用余弦夹角方式
	 * 
	 * @param doc1
	 * @param doc2
	 * @return
	 */
	public double computeSimilarity(int doc1, int doc2) {
		String cont1 = this.mapIdContent.get(doc1);
		String cont2 = this.mapIdContent.get(doc2);
		double sim = 0.0;

		List<String> wordsOne = Arrays.asList(cont1.split(" "));
		List<String> wordsTwo = Arrays.asList(cont2.split(" "));
		HashMap<String, Double> mapWordTfidf = new HashMap<String, Double>(50);// for
																				// words
																				// Two
		for (int i = 1; i < wordsTwo.size(); i++) {
			String[] item = wordsTwo.get(i).split(":");
			String key = item[0].trim();
			double value = Double.parseDouble(item[1].trim());
			mapWordTfidf.put(key, value);
		}

		double length1 = this.computeVecLength(cont1);
		double length2 = this.computeVecLength(cont2);

		for (int index1 = 1; index1 < wordsOne.size(); index1++) {
			String word = wordsOne.get(index1).split(":")[0];
			double value1 = Double.parseDouble(wordsOne.get(index1).split(":")[1]);
			if (mapWordTfidf.containsKey(word)) {
				double value2 = mapWordTfidf.get(word);
				sim += value1 * value2;
			}
		}

		return sim / (Math.sqrt(length1) * Math.sqrt(length2));// Consine
																// Measure
	}
	
	/**
	 * 采用heat kernel函数计算相似度，其中t=1，log-(doc1-doc2)^2
	 * @param doc1
	 * @param doc2
	 * @return
	 */
	public double computeSimilarityHeatKernel(int doc1, int doc2) {
		String cont1 = this.mapIdContent.get(doc1);
		String cont2 = this.mapIdContent.get(doc2);
		double len = 0.0;

		List<String> wordsOne = Arrays.asList(cont1.split(" "));
		List<String> wordsTwo = Arrays.asList(cont2.split(" "));
		HashMap<String, Double> mapWordTfidf = new HashMap<String, Double>(50);// for
																				// words
																				// Two
		for (int i = 1; i < wordsTwo.size(); i++) {
			String[] item = wordsTwo.get(i).split(":");
			String key = item[0].trim();
			double value = Double.parseDouble(item[1].trim());
			mapWordTfidf.put(key, value);
		}

//		double length1 = this.computeVecLength(cont1);
//		double length2 = this.computeVecLength(cont2);

		for (int index1 = 1; index1 < wordsOne.size(); index1++) {
			String word = wordsOne.get(index1).split(":")[0];
			double value1 = Double.parseDouble(wordsOne.get(index1).split(":")[1]);
			if (mapWordTfidf.containsKey(word)) {
				double value2 = mapWordTfidf.get(word);
				len += Math.pow((value1 - value2),2);
				mapWordTfidf.remove(word);
			}else{
				len += Math.pow((value1),2);
			}
		}
		
		for (Map.Entry<String, Double> entry : mapWordTfidf.entrySet()) {	 
		    len+=Math.pow(entry.getValue(),2);
		}
		
		return Math.pow(Math.E, (-len/0.01));
	}

	/**
	 * 计算向量的长度
	 * 
	 * @param content
	 * @return
	 */
	private double computeVecLength(String content) {
		double length = 0.0;
		String[] wordFreq = content.split(" ");

		for (int i = 1; i < wordFreq.length; i++) {
			double freq = Double.parseDouble(wordFreq[i].split(":")[1]);
			length += freq * freq;
		}

		return length;
	}

	public static void main(String[] args) {
		String path = "D:\\NSFC\\project\\data\\GoogleSnippets\\traintest.txt.bow";
		Docs2Net net = new Docs2Net(path);
		net.docs2NetTopk(path);
	}
}
