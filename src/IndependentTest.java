

public class IndependentTest {

	private static StdDataset datasets;
	private static String trainFile;
	private static String testFile;

	//configuration
	private static final int TESTNUM = 5;
	private static final int TOPTHREASHOLD = 20;

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
		testFile = dataset4;
		
		// Confirm input information
		System.out.println("******Input Information******\nTrain File:\t" + trainFile);
		System.out.println("Test File:\t" + (testFile == null?"---Not Use---":testFile));
		System.out.println("Network File:\t" + networkEnds +" | " + networkRels);
		System.out.println("*****************************");
		
		// Read input data
		datasets = new StdDataset(networkEnds,networkRels);
		datasets.readData(trainFile);
		datasets.readData(testFile);

		datasets.standardizeData();

		// Prepare Network
		RegNetsInstance[] ins = new RegNetsInstance[TESTNUM];
		Thread[] trd = new Thread[TESTNUM];
		
		//statistics
//		HashMap<Integer,Integer> frequency = new HashMap<Integer,Integer>();
		
		for(int i=0;i<TESTNUM;i++){
			ins[i] = new RegNetsInstance(
					datasets.getDatasets().get(0),
					datasets.getGroundTruths().get(0),
					datasets);
			ins[i].setHideIterMsg(false);

			System.out.println("Instance " + i + " training...");
			trd[i] = new Thread(ins[i]);
			trd[i].start();
//			System.out.println(re.output);
//			
//			ObjectWithValue[] ary = ins.getRecorder().ary;
//			for(int j=ary.length-1;j>ary.length - 2 - TOPTHREASHOLD;j--){
//				ObjectWithValue o = ary[j];
//				System.out.printf("%5d:%5.4f:%20s\n",(int)o.o,o.value,datasets.getHiddenNodeContent((int)o.o));
//			}
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
		
		
		for(int i=0;i<TESTNUM;i++){
			ins[i].test(
					datasets.getDatasets().get(1),
					datasets.getGroundTruths().get(1));
			Recorder re = ins[i].getRecorder();
			double useCutoff = re.useCutoff;
			int idx = re.cutoff.indexOf(useCutoff);
			double tpr = re.tpr.get(idx);
			double fpr = re.fpr.get(idx);
			double error = re.error;
			
			errorMean += error / TESTNUM;
			errorSD += error*error / TESTNUM;
			tprMean += tpr / TESTNUM;
			tprSD += tpr*tpr / TESTNUM;
			fprMean += fpr / TESTNUM;
			fprSD += fpr*fpr / TESTNUM;

			System.out.println(re.output);
		}
		errorSD = Math.sqrt(errorSD - errorMean*errorMean);
		tprSD = Math.sqrt(tprSD - tprMean*tprMean);
		fprSD = Math.sqrt(fprSD - fprMean*fprMean);
		
		System.out.printf("Error: %5.4f +- %5.4f\n",errorMean,errorSD);
		System.out.printf("Tpr: %5.4f +- %5.4f\n",tprMean,tprSD);
		System.out.printf("Fpr: %5.4f +- %5.4f\n",fprMean,fprSD);
		
	}
}
