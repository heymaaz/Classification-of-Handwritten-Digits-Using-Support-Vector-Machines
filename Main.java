import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

class Main {
    private static final String PATH_TO_CSV_DATASET1 = "cw2DataSet1.csv";
    private static final String PATH_TO_CSV_DATASET2 = "cw2DataSet2.csv";
    private static final int PIXELS_IN_ROW = 8;//8 pixels in a row
    private static final int PIXEL_COUNT = PIXELS_IN_ROW*PIXELS_IN_ROW;//8x8 pixels in the image = 64 pixels
    private static final int ROW_LENGTH = PIXEL_COUNT+1;//64 pixels + 1 label in the csv file for each row = 65 columns
    private static final int FILE_NOT_FOUND_CODE = -1;//Error code for file not found
    private static final int ERROR_READING_FILE_CODE = -2;//Error code for error reading file
	private static final double PIXEL_VALUE = 16.0;//16 different pixel values in the image

	private static final double GAMMA_VALUE = 0.5;//Gamma value for RBF kernel. It is the gamma parameter for the RBF kernel. The larger the value, the more complex the model. The smaller the value, the simpler the model.
	private static final double COST_PARAMETER = 1.0;//Cost parameter. It is the cost of misclassification. A large C gives you low bias and high variance. A small C gives you higher bias and lower variance.
	private static final double STOPPING_CRITERIA = 1e-3;//Stopping criteria. It is the tolerance of the termination criterion. The smaller the value, the more accurate the result and the longer it takes to converge.

	public static void main(String[] args) {
		// This is the main function
		// It runs the SVM model on both datasets and prints the average accuracy

		// Run the SVM model on both datasets
		double accuracy1 = runSVM(PATH_TO_CSV_DATASET1, PATH_TO_CSV_DATASET2);
		double accuracy2 = runSVM(PATH_TO_CSV_DATASET2, PATH_TO_CSV_DATASET1);

		// Print the accuracy of the model on both datasets
		System.out.println("Accuracy for model trained on dataset1 and tested on dataset2: " + accuracy1 + "%");
		System.out.println("Accuracy for model trained on dataset2 and tested on dataset1: " + accuracy2 + "%");
		
		// Print the average accuracy of the model
		System.out.println("Average Accuracy: " + (accuracy1 + accuracy2)/2 + "%");
	}

	static double runSVM(String trainFile, String testFile){
		// This function trains a SVM model with the given dataset trainFile
		// It returns the accuracy of the model on the testFile dataset

		// Initialize the SVM problem with dataset1
		svm_problem prob = initSVM(trainFile);

		// Initialize the SVM parameters
		svm_parameter param = initSVMParam();

		// Train the SVM model with dataset1
		svm_model model = trainSVM(prob, param);

		// Test the SVM model on dataset2
		return testSVM(model, testFile);

	}

	static svm_problem initSVM(String trainFile){
		// This function initializes the SVM problem with the given dataset trainFile
		// It returns the svm_problem object

		// Read the training data from the csv file
		int[][] trainData = readCSV(trainFile);
		
		// Initialize the svm_problem object
		svm_problem prob = new svm_problem();

		int dataCount = trainData.length; // Number of training data
		prob.labels = new double[dataCount]; // Target values
		prob.count = dataCount; // Number of training data
		prob.values = new svm_node[dataCount][PIXEL_COUNT]; // Feature nodes

		// Initialize the svm_problem object with the training data
		for(int image = 0; image < dataCount; image++) {//For each image in the training data
			for(int pixel = 0; pixel < PIXEL_COUNT; pixel++) {//For each pixel in a image
				prob.values[image][pixel] = new svm_node();//Create a new node for each pixel
				prob.values[image][pixel].index = pixel + 1;//Index of the pixel in the image is the pixel number + 1	as the index is 1-based
				prob.values[image][pixel].value = trainData[image][pixel] / PIXEL_VALUE; //  pixel values are in 0-16 so we normalize them to 0-1 by dividing by 16
			}
			prob.labels[image] = trainData[image][PIXEL_COUNT];
		}

		// return the svm_problem object
		return prob;
	}

	static svm_parameter initSVMParam(){
		// This function initializes the SVM parameters for the SVM model
		// It calls the setSVMParam function to set the SVM parameters
		// It returns the svm_parameter object

		// Create a new svm_parameter object
		svm_parameter param = setSVMParam(new svm_parameter());

		// return the svm_parameter object
		return param;
	}

	static svm_parameter setSVMParam(svm_parameter param){
		// This function sets the SVM parameters for the given svm_parameter object
		// It returns the svm_parameter object

		// Set SVM parameters
		param.svm_type = svm_parameter.C_SVC;//C-Support Vector Classification
		param.kernel_type = svm_parameter.RBF;//Radial Basis Function
		param.gamma = GAMMA_VALUE;//Gamma value for RBF kernel
		param.C = COST_PARAMETER;//Cost parameter
		param.eps = STOPPING_CRITERIA;//Stopping criteria
		
		// return the svm_parameter object
		return param;
	}

	static svm_model trainSVM(svm_problem prob, svm_parameter param){
		// This function trains the SVM model with the given svm_problem and svm_parameter
		// It returns the svm_model object

		// Train the SVM model
		svm_model model = svm.svm_train(prob, param);

		// return the svm_model object
		return model;
	}

	static double testSVM(svm_model model, String testFile){
		// This function tests the SVM model on the given testFile dataset
		// It calls the countCorrectCategorisation function to count the number of correct predictions by the model
		// It returns the accuracy of the model on the testFile dataset in percentage

		// Read the test data from the csv file
		int[][] testData = readCSV(testFile);

		return (double)countCorrectCategorisation(model,testData)/testData.length*100;//Return the accuracy of the model
	}

	static int countCorrectCategorisation(svm_model model, int[][] testData){
		// This function tests the SVM model on the given testFile dataset
		// It returns the number of correct predictions by the model

		// Initialize the correct counter for counting the number of correct predictions by the model to 0
		int correct = 0;

		// Test the SVM model on the test data
		for(int image = 0; image < testData.length; image++) {//For each image in the test data
			svm_node[] svmNodeArray = new svm_node[PIXEL_COUNT];//Create a svm_node array for each image
			for(int pixel = 0; pixel < PIXEL_COUNT; pixel++) {//For each pixel in an image
				svmNodeArray[pixel] = new svm_node();//Create a new node for each pixel
				svmNodeArray[pixel].index = pixel + 1;//Index of the pixel in the image is the pixel number + 1 as the index is 1-based
				svmNodeArray[pixel].value = testData[image][pixel] / PIXEL_VALUE; // pixel values are in 0-16 so we normalize them to 0-1 by dividing by 16
			}
			double label = svm.svm_predict(model, svmNodeArray);//Predict the label of the image
			if(label == testData[image][PIXEL_COUNT]) {//If the predicted label is equal to the actual label
				correct++;//Increment the correct counter
			}
		}
		return correct;
	}

	static int[][] readCSV(String filename) {
        // This function reads a csv file and returns a 2D array of integers containing the data from the csv file
        // If there is an error reading the file it prints the stack trace and returns null
        try{
            File myFile = new File(filename);//creates a new file instance
            int lineCount = countLines(myFile);
            if(lineCount == FILE_NOT_FOUND_CODE || lineCount == ERROR_READING_FILE_CODE){
                return null;
            }
            int[][] myArray = new int[lineCount][ROW_LENGTH];//Create an array to store the data from the csv file. Each row has 65 columns(64 pixels + 1 label)
            Scanner readMyFile = new Scanner(myFile);//creates a new scanner instance
            for (int i = 0; i < lineCount; i++) {
                String[] line = readMyFile.nextLine().split(",");
                for (int j = 0; j < 65; j++) {
                    myArray[i][j] = Integer.parseInt(line[j]);
                }
            }
            readMyFile.close();
            return myArray;
        } catch (FileNotFoundException invalidPathToFile) {
            System.out.println("File not found");
            invalidPathToFile.printStackTrace();
            return null;
        }
    }

    static int countLines(File myFile) {
        // This function counts the number of lines in a file
        // It returns FILE_NOT_FOUND_CODE if the file is not found
        // It returns ERROR_READING_FILE_CODE if there is an error reading the file
        try{
            Scanner readMyFile = new Scanner(myFile);
            int lineCount = 0;
            while(readMyFile.hasNextLine()) {
                lineCount++;
                readMyFile.nextLine();
            }
            readMyFile.close();
            return lineCount;
        } catch (FileNotFoundException invalidPathToFile) {
            System.out.println("File not found");
            return FILE_NOT_FOUND_CODE;
        }
    }

}