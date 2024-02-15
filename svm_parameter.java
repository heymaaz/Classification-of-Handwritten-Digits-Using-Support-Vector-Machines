public class svm_parameter
{
	/* svm_type */
	public static final int C_SVC = 0;

	/* kernel_type */
	public static final int RBF = 2;

	public int svm_type;
	public int kernel_type;
	public double gamma;	// for poly/rbf/sigmoid

	// these are for training only
	public double cache_size; // in MB
	public double eps;	// stopping criteria
	public double C;	// for C_SVC, EPSILON_SVR and NU_SVR
	public int nr_weight;		// for C_SVC
	public int[] weight_label;	// for C_SVC
	public double[] weight;		// for C_SVC
	public int shrinking;	// use the shrinking heuristics
	public int probability; // do probability estimates
	
	

}