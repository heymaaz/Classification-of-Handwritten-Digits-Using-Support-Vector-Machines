import java.util.Random;

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//


class Cache {
	private final int l;
	private long size;
	private final class head_t
	{
		head_t prev, next;	// a cicular list
		float[] data;
		int len;		// data[0,len) is cached in this entry
	}
	private final head_t[] head;
	private head_t lru_head;

	Cache(int l_, long size_)
	{
		l = l_;
		size = size_;
		head = new head_t[l];
		for(int i=0;i<l;i++) head[i] = new head_t();
		size /= 4;
		size -= l * (16/4);	// sizeof(head_t) == 16
		size = Math.max(size, 2* (long) l);  // cache must be large enough for two columns
		lru_head = new head_t();
		lru_head.next = lru_head.prev = lru_head;
	}

	private void lru_delete(head_t h)
	{
		// delete from current location
		h.prev.next = h.next;
		h.next.prev = h.prev;
	}

	private void lru_insert(head_t h)
	{
		// insert to last position
		h.next = lru_head;
		h.prev = lru_head.prev;
		h.prev.next = h;
		h.next.prev = h;
	}

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	// java: simulate pointer using single-element array
	int get_data(int index, float[][] data, int len)
	{
		head_t h = head[index];
		if(h.len > 0) lru_delete(h);
		int more = len - h.len;

		if(more > 0)
		{
			// free old space
			while(size < more)
			{
				head_t old = lru_head.next;
				lru_delete(old);
				size += old.len;
				old.data = null;
				old.len = 0;
			}

			// allocate new space
			float[] new_data = new float[len];
			if(h.data != null) System.arraycopy(h.data,0,new_data,0,h.len);
			h.data = new_data;
			size -= more;
			do {int tmp=h.len; h.len=len; len=tmp;} while(false);
		}

		lru_insert(h);
		data[0] = h.data;
		return len;
	}

	void swap_index(int i, int j)
	{
		if(i==j) return;

		if(head[i].len > 0) lru_delete(head[i]);
		if(head[j].len > 0) lru_delete(head[j]);
		do {float[] tmp=head[i].data; head[i].data=head[j].data; head[j].data=tmp;} while(false);
		do {int tmp=head[i].len; head[i].len=head[j].len; head[j].len=tmp;} while(false);
		if(head[i].len > 0) lru_insert(head[i]);
		if(head[j].len > 0) lru_insert(head[j]);

		if(i>j) do {int tmp=i; i=j; j=tmp;} while(false);
		for(head_t h = lru_head.next; h!=lru_head; h=h.next)
		{
			if(h.len > i)
			{
				if(h.len > j)
					do {float tmp=h.data[i]; h.data[i]=h.data[j]; h.data[j]=tmp;} while(false);
				else
				{
					// give up
					lru_delete(h);
					size += h.len;
					h.data = null;
					h.len = 0;
				}
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
abstract class QMatrix {
	abstract float[] get_Q(int column, int len);
	abstract double[] get_QD();
	abstract void swap_index(int i, int j);
};

abstract class Kernel extends QMatrix {
	private svm_node[][] x;
	private final double[] x_square;

	private final double gamma;

	abstract float[] get_Q(int column, int len);
	abstract double[] get_QD();

	void swap_index(int i, int j)
	{
		do {svm_node[] tmp=x[i]; x[i]=x[j]; x[j]=tmp;} while(false);
		if(x_square != null) do {double tmp=x_square[i]; x_square[i]=x_square[j]; x_square[j]=tmp;} while(false);
	}
	double kernel_function(int i, int j)
	{
		return Math.exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}

	Kernel(int l, svm_node[][] x_, svm_parameter param)
	{
		this.gamma = param.gamma;

		x = (svm_node[][])x_.clone();
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}

	static double dot(svm_node[] x, svm_node[] y)
	{
		double sum = 0;
		int xlen = x.length;
		int ylen = y.length;
		int i = 0;
		int j = 0;
		while(i < xlen && j < ylen)
		{
			if(x[i].index == y[j].index)
				sum += x[i++].value * y[j++].value;
			else
			{
				if(x[i].index > y[j].index)
					++j;
				else
					++i;
			}
		}
		return sum;
	}

	static double k_function(svm_node[] x, svm_node[] y, svm_parameter param)
	{
		double sum = 0;
		int xlen = x.length;
		int ylen = y.length;
		int i = 0;
		int j = 0;
		while(i < xlen && j < ylen)
		{
			if(x[i].index == y[j].index)
			{
				double d = x[i++].value - y[j++].value;
				sum += d*d;
			}
			else if(x[i].index > y[j].index)
			{
				sum += y[j].value * y[j].value;
				++j;
			}
			else
			{
				sum += x[i].value * x[i].value;
				++i;
			}
		}

		while(i < xlen)
		{
			sum += x[i].value * x[i].value;
			++i;
		}

		while(j < ylen)
		{
			sum += y[j].value * y[j].value;
			++j;
		}

		return Math.exp(-param.gamma*sum);
		
	}
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
	int active_size;
	byte[] y;
	double[] G;		// gradient of objective function
	static final byte LOWER_BOUND = 0;
	static final byte UPPER_BOUND = 1;
	static final byte FREE = 2;
	byte[] alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double[] alpha;
	QMatrix Q;
	double[] QD;
	double eps;
	double Cp,Cn;
	double[] p;
	int[] active_set;
	double[] G_bar;		// gradient, if we treat free variables as 0
	int l;
	boolean unshrink;	

	static final double INF = java.lang.Double.POSITIVE_INFINITY;

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	boolean is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	boolean is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	boolean is_free(int i) { return alpha_status[i] == FREE; }

	// java: information about solution except alpha,
	// because we cannot return multiple values otherwise...
	static class SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	}

	void swap_index(int i, int j)
	{
		Q.swap_index(i,j);
		do {byte tmp=y[i]; y[i]=y[j]; y[j]=tmp;} while(false);
		do {double tmp=G[i]; G[i]=G[j]; G[j]=tmp;} while(false);
		do {byte tmp=alpha_status[i]; alpha_status[i]=alpha_status[j]; alpha_status[j]=tmp;} while(false);
		do {double tmp=alpha[i]; alpha[i]=alpha[j]; alpha[j]=tmp;} while(false);
		do {double tmp=p[i]; p[i]=p[j]; p[j]=tmp;} while(false);
		do {int tmp=active_set[i]; active_set[i]=active_set[j]; active_set[j]=tmp;} while(false);
		do {double tmp=G_bar[i]; G_bar[i]=G_bar[j]; G_bar[j]=tmp;} while(false);
	}

	void reconstruct_gradient()
	{
		// reconstruct inactive elements of G from G_bar and free variables

		if(active_size == l) return;

		int i,j;
		int nr_free = 0;

		for(j=active_size;j<l;j++)
			G[j] = G_bar[j] + p[j];

		for(j=0;j<active_size;j++)
			if(is_free(j))
				nr_free++;

		if(2*nr_free < active_size)
			svm.info("\nWARNING: using -h 0 may be faster\n");

		if (nr_free*l > 2*active_size*(l-active_size))
		{
			for(i=active_size;i<l;i++)
			{
				float[] Q_i = Q.get_Q(i,active_size);
				for(j=0;j<active_size;j++)
					if(is_free(j))
						G[i] += alpha[j] * Q_i[j];
			}
		}
		else
		{
			for(i=0;i<active_size;i++)
				if(is_free(i))
				{
					float[] Q_i = Q.get_Q(i,l);
					double alpha_i = alpha[i];
					for(j=active_size;j<l;j++)
						G[j] += alpha_i * Q_i[j];
				}
		}
	}

	void Solve(int l, QMatrix Q, double[] p_, byte[] y_,
		   double[] alpha_, double Cp, double Cn, double eps, SolutionInfo si, int shrinking)
	{
		this.l = l;
		this.Q = Q;
		QD = Q.get_QD();
		p = (double[])p_.clone();
		y = (byte[])y_.clone();
		alpha = (double[])alpha_.clone();
		this.Cp = Cp;
		this.Cn = Cn;
		this.eps = eps;
		this.unshrink = false;

		// initialize alpha_status
		{
			alpha_status = new byte[l];
			for(int i=0;i<l;i++)
				update_alpha_status(i);
		}

		// initialize active set (for shrinking)
		{
			active_set = new int[l];
			for(int i=0;i<l;i++)
				active_set[i] = i;
			active_size = l;
		}

		// initialize gradient
		{
			G = new double[l];
			G_bar = new double[l];
			int i;
			for(i=0;i<l;i++)
			{
				G[i] = p[i];
				G_bar[i] = 0;
			}
			for(i=0;i<l;i++)
				if(!is_lower_bound(i))
				{
					float[] Q_i = Q.get_Q(i,l);
					double alpha_i = alpha[i];
					int j;
					for(j=0;j<l;j++)
						G[j] += alpha_i*Q_i[j];
					if(is_upper_bound(i))
						for(j=0;j<l;j++)
							G_bar[j] += get_C(i) * Q_i[j];
				}
		}

		// optimization step

		int iter = 0;
		int max_iter = Math.max(10000000, l>Integer.MAX_VALUE/100 ? Integer.MAX_VALUE : 100*l);
		int counter = Math.min(l,1000)+1;
		int[] working_set = new int[2];

		while(iter < max_iter)
		{
			// show progress and do shrinking

			if(--counter == 0)
			{
				counter = Math.min(l,1000);
				if(shrinking!=0) do_shrinking();
				svm.info(".");
			}

			if(select_working_set(working_set)!=0)
			{
				// reconstruct the whole gradient
				reconstruct_gradient();
				// reset active set size and check
				active_size = l;
				svm.info("*");
				if(select_working_set(working_set)!=0)
					break;
				else
					counter = 1;	// do shrinking next iteration
			}

			int i = working_set[0];
			int j = working_set[1];

			++iter;

			// update alpha[i] and alpha[j], handle bounds carefully

			float[] Q_i = Q.get_Q(i,active_size);
			float[] Q_j = Q.get_Q(j,active_size);

			double C_i = get_C(i);
			double C_j = get_C(j);

			double old_alpha_i = alpha[i];
			double old_alpha_j = alpha[j];

			if(y[i]!=y[j])
			{
				double quad_coef = QD[i]+QD[j]+2*Q_i[j];
				if (quad_coef <= 0)
					quad_coef = 1e-12;
				double delta = (-G[i]-G[j])/quad_coef;
				double diff = alpha[i] - alpha[j];
				alpha[i] += delta;
				alpha[j] += delta;

				if(diff > 0)
				{
					if(alpha[j] < 0)
					{
						alpha[j] = 0;
						alpha[i] = diff;
					}
				}
				else
				{
					if(alpha[i] < 0)
					{
						alpha[i] = 0;
						alpha[j] = -diff;
					}
				}
				if(diff > C_i - C_j)
				{
					if(alpha[i] > C_i)
					{
						alpha[i] = C_i;
						alpha[j] = C_i - diff;
					}
				}
				else
				{
					if(alpha[j] > C_j)
					{
						alpha[j] = C_j;
						alpha[i] = C_j + diff;
					}
				}
			}
			else
			{
				double quad_coef = QD[i]+QD[j]-2*Q_i[j];
				if (quad_coef <= 0)
					quad_coef = 1e-12;
				double delta = (G[i]-G[j])/quad_coef;
				double sum = alpha[i] + alpha[j];
				alpha[i] -= delta;
				alpha[j] += delta;

				if(sum > C_i)
				{
					if(alpha[i] > C_i)
					{
						alpha[i] = C_i;
						alpha[j] = sum - C_i;
					}
				}
				else
				{
					if(alpha[j] < 0)
					{
						alpha[j] = 0;
						alpha[i] = sum;
					}
				}
				if(sum > C_j)
				{
					if(alpha[j] > C_j)
					{
						alpha[j] = C_j;
						alpha[i] = sum - C_j;
					}
				}
				else
				{
					if(alpha[i] < 0)
					{
						alpha[i] = 0;
						alpha[j] = sum;
					}
				}
			}

			// update G

			double delta_alpha_i = alpha[i] - old_alpha_i;
			double delta_alpha_j = alpha[j] - old_alpha_j;

			for(int k=0;k<active_size;k++)
			{
				G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
			}

			// update alpha_status and G_bar

			{
				boolean ui = is_upper_bound(i);
				boolean uj = is_upper_bound(j);
				update_alpha_status(i);
				update_alpha_status(j);
				int k;
				if(ui != is_upper_bound(i))
				{
					Q_i = Q.get_Q(i,l);
					if(ui)
						for(k=0;k<l;k++)
							G_bar[k] -= C_i * Q_i[k];
					else
						for(k=0;k<l;k++)
							G_bar[k] += C_i * Q_i[k];
				}

				if(uj != is_upper_bound(j))
				{
					Q_j = Q.get_Q(j,l);
					if(uj)
						for(k=0;k<l;k++)
							G_bar[k] -= C_j * Q_j[k];
					else
						for(k=0;k<l;k++)
							G_bar[k] += C_j * Q_j[k];
				}
			}

		}

		if(iter >= max_iter)
		{
			if(active_size < l)
			{
				// reconstruct the whole gradient to calculate objective value
				reconstruct_gradient();
				active_size = l;
				svm.info("*");
			}
			System.err.print("\nWARNING: reaching max number of iterations\n");
		}

		// calculate rho

		si.rho = calculate_rho();

		// calculate objective value
		{
			double v = 0;
			int i;
			for(i=0;i<l;i++)
				v += alpha[i] * (G[i] + p[i]);

			si.obj = v/2;
		}

		// put back the solution
		{
			for(int i=0;i<l;i++)
				alpha_[active_set[i]] = alpha[i];
		}

		si.upper_bound_p = Cp;
		si.upper_bound_n = Cn;

		svm.info("\noptimization finished, #iter = "+iter+"\n");
	}

	// return 1 if already optimal, return 0 otherwise
	int select_working_set(int[] working_set)
	{
		// return i,j such that
		// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
		// j: minimizes the decrease of obj value
		//    (if quadratic coefficeint <= 0, replace it with tau)
		//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

		double Gmax = -INF;
		double Gmax2 = -INF;
		int Gmax_idx = -1;
		int Gmin_idx = -1;
		double obj_diff_min = INF;

		for(int t=0;t<active_size;t++)
			if(y[t]==+1)
			{
				if(!is_upper_bound(t))
					if(-G[t] >= Gmax)
					{
						Gmax = -G[t];
						Gmax_idx = t;
					}
			}
			else
			{
				if(!is_lower_bound(t))
					if(G[t] >= Gmax)
					{
						Gmax = G[t];
						Gmax_idx = t;
					}
			}

		int i = Gmax_idx;
		float[] Q_i = null;
		if(i != -1) // null Q_i not accessed: Gmax=-INF if i=-1
			Q_i = Q.get_Q(i,active_size);

		for(int j=0;j<active_size;j++)
		{
			if(y[j]==+1)
			{
				if (!is_lower_bound(j))
				{
					double grad_diff=Gmax+G[j];
					if (G[j] >= Gmax2)
						Gmax2 = G[j];
					if (grad_diff > 0)
					{
						double obj_diff;
						double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
						if (quad_coef > 0)
							obj_diff = -(grad_diff*grad_diff)/quad_coef;
						else
							obj_diff = -(grad_diff*grad_diff)/1e-12;

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx=j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
			else
			{
				if (!is_upper_bound(j))
				{
					double grad_diff= Gmax-G[j];
					if (-G[j] >= Gmax2)
						Gmax2 = -G[j];
					if (grad_diff > 0)
					{
						double obj_diff;
						double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
						if (quad_coef > 0)
							obj_diff = -(grad_diff*grad_diff)/quad_coef;
						else
							obj_diff = -(grad_diff*grad_diff)/1e-12;

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx=j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
		}

		if(Gmax+Gmax2 < eps || Gmin_idx == -1)
			return 1;

		working_set[0] = Gmax_idx;
		working_set[1] = Gmin_idx;
		return 0;
	}

	private boolean be_shrunk(int i, double Gmax1, double Gmax2)
	{
		if(is_upper_bound(i))
		{
			if(y[i]==+1)
				return(-G[i] > Gmax1);
			else
				return(-G[i] > Gmax2);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				return(G[i] > Gmax2);
			else
				return(G[i] > Gmax1);
		}
		else
			return(false);
	}

	void do_shrinking()
	{
		int i;
		double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
		double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

		// find maximal violating pair first
		for(i=0;i<active_size;i++)
		{
			if(y[i]==+1)
			{
				if(!is_upper_bound(i))
				{
					if(-G[i] >= Gmax1)
						Gmax1 = -G[i];
				}
				if(!is_lower_bound(i))
				{
					if(G[i] >= Gmax2)
						Gmax2 = G[i];
				}
			}
			else
			{
				if(!is_upper_bound(i))
				{
					if(-G[i] >= Gmax2)
						Gmax2 = -G[i];
				}
				if(!is_lower_bound(i))
				{
					if(G[i] >= Gmax1)
						Gmax1 = G[i];
				}
			}
		}

		if(unshrink == false && Gmax1 + Gmax2 <= eps*10)
		{
			unshrink = true;
			reconstruct_gradient();
			active_size = l;
			svm.info("*");
		}

		for(i=0;i<active_size;i++)
			if (be_shrunk(i, Gmax1, Gmax2))
			{
				active_size--;
				while (active_size > i)
				{
					if (!be_shrunk(active_size, Gmax1, Gmax2))
					{
						swap_index(i,active_size);
						break;
					}
					active_size--;
				}
			}
	}

	double calculate_rho()
	{
		double r;
		int nr_free = 0;
		double ub = INF, lb = -INF, sum_free = 0;
		for(int i=0;i<active_size;i++)
		{
			double yG = y[i]*G[i];

			if(is_upper_bound(i))
			{
				if(y[i] < 0)
					ub = Math.min(ub,yG);
				else
					lb = Math.max(lb,yG);
			}
			else if(is_lower_bound(i))
			{
				if(y[i] > 0)
					ub = Math.min(ub,yG);
				else
					lb = Math.max(lb,yG);
			}
			else
			{
				++nr_free;
				sum_free += yG;
			}
		}

		if(nr_free>0)
			r = sum_free/nr_free;
		else
			r = (ub+lb)/2;

		return r;
	}

}

//
// Q matrices for various formulations
//
class SVC_Q extends Kernel
{
	private final byte[] y;
	private final Cache cache;
	private final double[] QD;

	SVC_Q(svm_problem prob, svm_parameter param, byte[] y_)
	{
		super(prob.count, prob.values, param);
		y = (byte[])y_.clone();
		cache = new Cache(prob.count,(long)(param.cache_size*(1<<20)));
		QD = new double[prob.count];
		for(int i=0;i<prob.count;i++)
			QD[i] = kernel_function(i,i);
	}

	float[] get_Q(int i, int len)
	{
		float[][] data = new float[1][];
		int start, j;
		if((start = cache.get_data(i,data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[0][j] = (float)(y[i]*y[j]*kernel_function(i,j));
		}
		return data[0];
	}

	double[] get_QD()
	{
		return QD;
	}

	void swap_index(int i, int j)
	{
		cache.swap_index(i,j);
		super.swap_index(i,j);
		do {byte tmp=y[i]; y[i]=y[j]; y[j]=tmp;} while(false);
		do {double tmp=QD[i]; QD[i]=QD[j]; QD[j]=tmp;} while(false);
	}
}

public class svm {
	//
	// construct and solve various formulations
	//
	public static final Random rand = new Random();

	private static svm_print_interface svm_print_stdout = new svm_print_interface()
	{
		public void print(String s)
		{
			System.out.print(s);
			System.out.flush();
		}
	};

	private static svm_print_interface svm_print_string = svm_print_stdout;

	static void info(String s)
	{
		//svm_print_string.print(s);
	}

	private static void solve_c_svc(svm_problem prob, svm_parameter param,
					double[] alpha, Solver.SolutionInfo si,
					double Cp, double Cn)
	{
		int l = prob.count;
		double[] minus_ones = new double[l];
		byte[] y = new byte[l];

		int i;

		for(i=0;i<l;i++)
		{
			alpha[i] = 0;
			minus_ones[i] = -1;
			if(prob.labels[i] > 0) y[i] = +1; else y[i] = -1;
		}

		Solver s = new Solver();
		s.Solve(l, new SVC_Q(prob,param,y), minus_ones, y,
			alpha, Cp, Cn, param.eps, si, param.shrinking);

		double sum_alpha=0;
		for(i=0;i<l;i++)
			sum_alpha += alpha[i];

		if (Cp==Cn)
			svm.info("nu = "+sum_alpha/(Cp*prob.count)+"\n");

		for(i=0;i<l;i++)
			alpha[i] *= y[i];
	}
	
	//
	// decision_function
	//
	static class decision_function
	{
		double[] alpha;
		double rho;
	};

	static decision_function svm_train_one(
		svm_problem prob, svm_parameter param,
		double Cp, double Cn)
	{
		double[] alpha = new double[prob.count];
		Solver.SolutionInfo si = new Solver.SolutionInfo();
		switch(param.svm_type)
		{
			case svm_parameter.C_SVC:
				solve_c_svc(prob,param,alpha,si,Cp,Cn);
				break;
			
		}

		svm.info("obj = "+si.obj+", rho = "+si.rho+"\n");

		// output SVs

		int nSV = 0;
		int nBSV = 0;
		for(int i=0;i<prob.count;i++)
		{
			if(Math.abs(alpha[i]) > 0)
			{
				++nSV;
				if(prob.labels[i] > 0)
				{
					if(Math.abs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
				}
				else
				{
					if(Math.abs(alpha[i]) >= si.upper_bound_n)
						++nBSV;
				}
			}
		}

		svm.info("nSV = "+nSV+", nBSV = "+nBSV+"\n");

		decision_function f = new decision_function();
		f.alpha = alpha;
		f.rho = si.rho;
		return f;
	}

	
	// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
	// perm, length l, must be allocated before calling this subroutine
	private static void svm_group_classes(svm_problem prob, int[] nr_class_ret, int[][] label_ret, int[][] start_ret, int[][] count_ret, int[] perm)
	{
		int l = prob.count;
		int max_nr_class = 16;
		int nr_class = 0;
		int[] label = new int[max_nr_class];
		int[] count = new int[max_nr_class];
		int[] data_label = new int[l];
		int i;

		for(i=0;i<l;i++)
		{
			int this_label = (int)(prob.labels[i]);
			int j;
			for(j=0;j<nr_class;j++)
			{
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			}
			data_label[i] = j;
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					int[] new_data = new int[max_nr_class];
					System.arraycopy(label,0,new_data,0,label.length);
					label = new_data;
					new_data = new int[max_nr_class];
					System.arraycopy(count,0,new_data,0,count.length);
					count = new_data;
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		//
		// Labels are ordered by their first occurrence in the training set.
		// However, for two-class sets with -1/+1 labels and -1 appears first,
		// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
		//
		if (nr_class == 2 && label[0] == -1 && label[1] == +1)
		{
			do {int tmp=label[0]; label[0]=label[1]; label[1]=tmp;} while(false);
			do {int tmp=count[0]; count[0]=count[1]; count[1]=tmp;} while(false);
			for(i=0;i<l;i++)
			{
				if(data_label[i] == 0)
					data_label[i] = 1;
				else
					data_label[i] = 0;
			}
		}

		int[] start = new int[nr_class];
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+count[i-1];
		for(i=0;i<l;i++)
		{
			perm[start[data_label[i]]] = i;
			++start[data_label[i]];
		}
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+count[i-1];

		nr_class_ret[0] = nr_class;
		label_ret[0] = label;
		start_ret[0] = start;
		count_ret[0] = count;
	}

	
	//
	// Interface functions
	//
	public static svm_model svm_train(svm_problem prob, svm_parameter param)
	{
		svm_model model = new svm_model();
		model.param = param;
		
		// classification
		int l = prob.count;
		int[] tmp_nr_class = new int[1];
		int[][] tmp_label = new int[1][];
		int[][] tmp_start = new int[1][];
		int[][] tmp_count = new int[1][];
		int[] perm = new int[l];

		// group training data of the same class
		svm_group_classes(prob,tmp_nr_class,tmp_label,tmp_start,tmp_count,perm);
		int nr_class = tmp_nr_class[0];
		int[] label = tmp_label[0];
		int[] start = tmp_start[0];
		int[] count = tmp_count[0];

		if(nr_class == 1)
			svm.info("WARNING: training data in only one class. See README for details.\n");

		svm_node[][] x = new svm_node[l][];
		int i;
		for(i=0;i<l;i++)
			x[i] = prob.values[perm[i]];

		// calculate weighted C

		double[] weighted_C = new double[nr_class];
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param.C;
		for(i=0;i<param.nr_weight;i++)
		{
			int j;
			for(j=0;j<nr_class;j++)
				if(param.weight_label[i] == label[j])
					break;
			if(j == nr_class)
				System.err.print("WARNING: class label "+param.weight_label[i]+" specified in weight is not found\n");
			else
				weighted_C[j] *= param.weight[i];
		}

		// train k*(k-1)/2 models

		boolean[] nonzero = new boolean[l];
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function[] f = new decision_function[nr_class*(nr_class-1)/2];

		double[] probA=null,probB=null;
		if (param.probability == 1)
		{
			probA=new double[nr_class*(nr_class-1)/2];
			probB=new double[nr_class*(nr_class-1)/2];
		}

		int p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob = new svm_problem();
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.count = ci+cj;
				sub_prob.values = new svm_node[sub_prob.count][];
				sub_prob.labels = new double[sub_prob.count];
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.values[k] = x[si+k];
					sub_prob.labels[k] = +1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.values[ci+k] = x[sj+k];
					sub_prob.labels[ci+k] = -1;
				}

				if(param.probability == 1)
				{
					double[] probAB=new double[2];
					//svm_binary_svc_probability(sub_prob,param,weighted_C[i],weighted_C[j],probAB);
					probA[p]=probAB[0];
					probB[p]=probAB[1];
				}

				f[p] = svm_train_one(sub_prob,param,weighted_C[i],weighted_C[j]);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && Math.abs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && Math.abs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				++p;
			}

		// build output

		model.nr_class = nr_class;

		model.label = new int[nr_class];
		for(i=0;i<nr_class;i++)
			model.label[i] = label[i];

		model.rho = new double[nr_class*(nr_class-1)/2];
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model.rho[i] = f[i].rho;

		if(param.probability == 1)
		{
			model.probA = new double[nr_class*(nr_class-1)/2];
			model.probB = new double[nr_class*(nr_class-1)/2];
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model.probA[i] = probA[i];
				model.probB[i] = probB[i];
			}
		}
		else
		{
			model.probA=null;
			model.probB=null;
		}
		model.prob_density_marks = null;	// for one-class SVM probabilistic outputs only

		int total_sv = 0;
		int[] nz_count = new int[nr_class];
		model.nSV = new int[nr_class];
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{
					++nSV;
					++total_sv;
				}
			model.nSV[i] = nSV;
			nz_count[i] = nSV;
		}

		svm.info("Total nSV = "+total_sv+"\n");

		model.l = total_sv;
		model.SV = new svm_node[total_sv][];
		model.sv_indices = new int[total_sv];
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i])
			{
				model.SV[p] = x[i];
				model.sv_indices[p++] = perm[i] + 1;
			}

		int[] nz_start = new int[nr_class];
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model.sv_coef = new double[nr_class-1][];
		for(i=0;i<nr_class-1;i++)
			model.sv_coef[i] = new double[total_sv];

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];

				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model.sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model.sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}
		
		return model;
	}

	public static double svm_predict_values(svm_model model, svm_node[] x, double[] dec_values)
	{
		int i;
		
		int nr_class = model.nr_class;
		int l = model.l;

		double[] kvalue = new double[l];
		for(i=0;i<l;i++)
			kvalue[i] = Kernel.k_function(x,model.SV[i],model.param);

		int[] start = new int[nr_class];
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model.nSV[i-1];

		int[] vote = new int[nr_class];
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model.nSV[i];
				int cj = model.nSV[j];

				int k;
				double[] coef1 = model.sv_coef[j-1];
				double[] coef2 = model.sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model.rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		return model.label[vote_max_idx];
	}

	public static double svm_predict(svm_model model, svm_node[] x)
	{
		int nr_class = model.nr_class;
		double[] dec_values;
		dec_values = new double[nr_class*(nr_class-1)/2];
		return svm_predict_values(model, x, dec_values);
	}

}