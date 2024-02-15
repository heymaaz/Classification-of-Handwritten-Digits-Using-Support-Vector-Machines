public class svm_parameter
{
	/* svm_type */
	public static final int C_SVC = 0;

	/* kernel_type */
	public static final int RBF = 2;

	public int svm_type;
	public int kernel_type;
	public double gamma;	

	// these are for training only
	public double cache_size; // in MB
	public double eps;	// stopping criteria
	public double C;	
	public int nr_weight;		
	public int[] weight_label;	
	public double[] weight;		
	public int shrinking;	// use the shrinking heuristics
	public int probability; // do probability estimates
	
	

}