import java.util.ArrayList;
import java.util.Random;

public class CrossValidationTest {

	private static StdDataset datasets;
	private static String trainFile;

	//configuration
	private static final int TESTNUM = 5;
	private static final int TOPTHREASHOLD = 20;

	public CrossValidationTest() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) {
		// Default value
		String networkEnds = "kb/BEL_LargeCorpus.ents";
		String networkRels = "kb/BEL_LargeCorpus.rels";
		
		String dataset1 = "UC/GSE12251/GSE12251_data.txt";
		String dataset2 = "UC/GSE14580/GSE14580_data.txt";
		String dataset3 = "Rejection/GSE21374/GSE21374_data.txt";
		String dataset4 = "Rejection/GSE50058/GSE50058_data.txt";
		
		//select test dataset		
		trainFile = dataset3;
		
		// Confirm input information
		System.out.println("******Input Information******\nTrain File:\t" + trainFile);
		System.out.println("Network File:\t" + networkEnds +" | " + networkRels);
		System.out.println("*****************************");

		// Read input data
		datasets = new StdDataset(networkEnds,networkRels);
		datasets.readData(trainFile);

		datasets.standardizeData();

		// Prepare Network
		RegNetsInstance[] ins = new RegNetsInstance[TESTNUM];
		Thread[] trd = new Thread[TESTNUM];

		//used data
		ArrayList<ArrayList<Double>> d1 = new ArrayList<ArrayList<Double>>(datasets.getDatasets().get(0));
		ArrayList<Integer> gt1 = new ArrayList<Integer>(datasets.getGroundTruths().get(0));
		ArrayList<ArrayList<ArrayList<Double>>> dt = new ArrayList<ArrayList<ArrayList<Double>>>();
		ArrayList<ArrayList<Integer>> gtt = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> pool = new ArrayList<Integer>();
		for(int i=0;i<TESTNUM;i++){
			dt.add(new ArrayList<ArrayList<Double>>());
			gtt.add(new ArrayList<Integer>());
			pool.add(i);
		}
		int n = d1.size();
		Random rnd = new Random();
		for(int i=0;i<n;i++){
			int rNum = rnd.nextInt(pool.size());
			int idx = pool.get(rNum);
			dt.get(idx).add(d1.get(i));
			gtt.get(idx).add(gt1.get(i));
			if(dt.get(idx).size()>n/5){
				pool.remove(rNum);
			}
		}
		for(int i=0;i<TESTNUM;i++){
			ArrayList<ArrayList<Double>> trainD = new ArrayList<ArrayList<Double>>();
			ArrayList<Integer> trainGt = new ArrayList<Integer>();
			ArrayList<ArrayList<Double>> testD = new ArrayList<ArrayList<Double>>();
			ArrayList<Integer> testGt = new ArrayList<Integer>();
			for(int j=0;j<TESTNUM;j++){
				if(i==j){
					testD.addAll(dt.get(j));
					testGt.addAll(gtt.get(j));
				}else{
					trainD.addAll(dt.get(j));
					trainGt.addAll(gtt.get(j));					
				}
			}
			ins[i] = new RegNetsInstance(
					trainD,
					trainGt,
					testD,
					testGt,
					datasets);
			ins[i].setHideIterMsg(false);

			System.out.println("Instance " + i + " training...");
			trd[i] = new Thread(ins[i]);
			trd[i].start();
		}
		
		try {
			for(int i=0;i<TESTNUM;i++){
				trd[i].join();
				System.out.println(i+" joined");
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		double errorMean = 0;
		double errorSD = 0;
		double tprMean = 0;
		double tprSD = 0;
		double fprMean = 0;
		double fprSD = 0;
		double bAccMean = 0;
		double bAccSD = 0;
		
		
		for(int i=0;i<TESTNUM;i++){
			ins[i].test(
					dt.get(i),
					gtt.get(i));
			Recorder re = ins[i].getRecorder();
			double useCutoff = re.useCutoff;
			int idx = re.cutoff.indexOf(useCutoff);
			double tpr = re.tpr.get(idx);
			double fpr = re.fpr.get(idx);
			double error = re.error;
			double bAcc = (re.sensitivity + re.specificity)/2;
			
			errorMean += error / TESTNUM;
			errorSD += error*error / TESTNUM;
			tprMean += tpr / TESTNUM;
			tprSD += tpr*tpr / TESTNUM;
			fprMean += fpr / TESTNUM;
			fprSD += fpr*fpr / TESTNUM;
			bAccMean += bAcc / TESTNUM;
			bAccSD += bAcc*bAcc / TESTNUM;
			
			System.out.println(re.output);
		}
		errorSD = Math.sqrt(errorSD - errorMean*errorMean);
		tprSD = Math.sqrt(tprSD - tprMean*tprMean);
		fprSD = Math.sqrt(fprSD - fprMean*fprMean);
		bAccSD = Math.sqrt(bAccSD - bAccMean*bAccMean);
		
		
		System.out.printf("Error: %5.4f +- %5.4f\n",errorMean,errorSD);
		System.out.printf("Tpr: %5.4f +- %5.4f\n",tprMean,tprSD);
		System.out.printf("Fpr: %5.4f +- %5.4f\n",fprMean,fprSD);
		System.out.printf("Balanced Accuracy: %5.4f +- %5.4f\n",bAccMean,bAccSD);
	}
}
