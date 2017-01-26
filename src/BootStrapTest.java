import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

public class BootStrapTest {

	private static StdDataset datasets;
	private static String useDataset;
	
	//configuration
	private static final int BOOTSTRAPNUM = 100 ;
	private static final int BOOTSTRAPTOPTHREASHOLD1 = 10;
	private static final int BOOTSTRAPTOPTHREASHOLD2 = 30;
	private static final int BOOTSTRAPTOPTHREASHOLD3 = 100;

	public static void main(String[] args) {		
		// Default value
		String networkEnds = "kb/BEL_LargeCorpus.ents";
		String networkRels = "kb/BEL_LargeCorpus.rels";
		
		String dataset1 = "UC/GSE12251/GSE12251_data.txt";
		String dataset2 = "UC/GSE14580/GSE14580_data.txt";
		String dataset3 = "Rejection/GSE21374/GSE21374_data.txt";
		String dataset4 = "Rejection/GSE50058/GSE50058_data.txt";
		
		//select test dataset
		useDataset = dataset2;
		
		// Confirm input information
		System.out.println("******Input Information******\nTrain File:\t" + useDataset);
		System.out.println("Network File:\t" + networkEnds +" | " + networkRels);
		System.out.println("*****************************");
		
		// Read input data
		datasets = new StdDataset(networkEnds,networkRels);
		datasets.readData(useDataset);

		datasets.standardizeData();

		// Prepare Network
		RegNetsInstance ins;
		
		//used data
		ArrayList<ArrayList<Double>> d0 = datasets.getDatasets().get(0);
		ArrayList<Integer> g0 = new ArrayList<Integer>(datasets.getGroundTruths().get(0));
		
		//statistics
		HashMap<Integer,Integer> frequency1 = new HashMap<Integer,Integer>();
		HashMap<Integer,Integer> frequency2 = new HashMap<Integer,Integer>();
		HashMap<Integer,Integer> frequency3 = new HashMap<Integer,Integer>();
		
		
		//bootstrap
		for(int i=0;i<BOOTSTRAPNUM;i++){
			int n = d0.size();
			ArrayList<ArrayList<Double>> trainSet = new ArrayList<ArrayList<Double>>();
			ArrayList<Integer> trainGt = new ArrayList<Integer>();
			ArrayList<ArrayList<Double>> validSet = new ArrayList<ArrayList<Double>>();
			ArrayList<Integer> validGt = new ArrayList<Integer>();
			
			HashSet<Integer> hs = new HashSet<Integer>();
			for(int j=0;j<n;j++){
				int idx = (int) (Math.random()*n);
				trainSet.add(d0.get(idx));
				trainGt.add(g0.get(idx));
				hs.add(idx);
			}
			for(int j=0;j<n;j++){
				if(!hs.contains(j)){
					validSet.add(d0.get(j));
					validGt.add(g0.get(j));
				}
			}
			
			ins = new RegNetsInstance(
					trainSet,
					trainGt,
					validSet,
					validGt,
					datasets);
			System.out.print("Instance " + i + " training("+hs.size()+"/"+d0.size()+")...");
			ins.runtest(i);
			ins.calcResult();
			
			ObjectWithValue[] ary = ins.getRecorder().ary;
			for(int j=ary.length-1;j>ary.length-2-BOOTSTRAPTOPTHREASHOLD1;j--){
				ObjectWithValue o = ary[j];
				if(frequency1.containsKey(o.o)){
					frequency1.put((Integer) o.o, frequency1.get(o.o)+1);
				}else{
					frequency1.put((Integer) o.o, 1);
				}
			}
			for(int j=ary.length-1;j>ary.length-2-BOOTSTRAPTOPTHREASHOLD2;j--){
				ObjectWithValue o = ary[j];
				if(frequency2.containsKey(o.o)){
					frequency2.put((Integer) o.o, frequency2.get(o.o)+1);
				}else{
					frequency2.put((Integer) o.o, 1);
				}
			}
			for(int j=ary.length-1;j>ary.length-2-BOOTSTRAPTOPTHREASHOLD3;j--){
				ObjectWithValue o = ary[j];
				if(frequency3.containsKey(o.o)){
					frequency3.put((Integer) o.o, frequency3.get(o.o)+1);
				}else{
					frequency3.put((Integer) o.o, 1);
				}
			}

			System.out.println(" Done.");
		}
		
		ObjectWithValue[]  ary;
		int j;
		
		ary = new ObjectWithValue[frequency1.size()];
		j=0;
        for(Integer i:frequency1.keySet()){
        	ary[j++] = new ObjectWithValue(i,frequency1.get(i));
        }
        Arrays.sort(ary);		
        for(int i= ary.length>300?ary.length-300:0;i<ary.length;i++){
        	ObjectWithValue o = ary[i];
        	System.out.printf("%5d:%5d:%20s\n",(int)o.o,(int)o.value,datasets.getHiddenNodeContent((int)o.o));
        }
        System.out.println("------");
        
		ary = new ObjectWithValue[frequency2.size()];
		j=0;
        for(Integer i:frequency2.keySet()){
        	ary[j++] = new ObjectWithValue(i,frequency2.get(i));
        }
        Arrays.sort(ary);		
        for(int i= ary.length>300?ary.length-300:0;i<ary.length;i++){
        	ObjectWithValue o = ary[i];
        	System.out.printf("%5d:%5d:%20s\n",(int)o.o,(int)o.value,datasets.getHiddenNodeContent((int)o.o));
        }
        System.out.println("------");
        
		ary = new ObjectWithValue[frequency3.size()];
		j=0;
        for(Integer i:frequency3.keySet()){
        	ary[j++] = new ObjectWithValue(i,frequency3.get(i));
        }
        Arrays.sort(ary);		
        for(int i= ary.length>300?ary.length-300:0;i<ary.length;i++){
        	ObjectWithValue o = ary[i];
        	System.out.printf("%5d:%5d:%20s\n",(int)o.o,(int)o.value,datasets.getHiddenNodeContent((int)o.o));
        }
        System.out.println("------");
	}

}
