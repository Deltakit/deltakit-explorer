import traceback
from deltakit_explorer.analysis.threshold import ThresholdEstimator
from deltakit_explorer.plotting.threshold_plot import create_threshold_plot
from deltakit_explorer.qpu import ToyNoise

def main():
    try:
        print("Starting simulation")
        
        estimator = ThresholdEstimator(
            num_shots=1000, 
            precision=0.005, 
            noise_model_class=ToyNoise
        )

        results = estimator.run_parallel_searches([(3, 5), (5, 7)])

        data = estimator.get_search_history()
        avg_threshold = sum(results.values()) / len(results)
        
        final_path = "threshold_plot_full.png"
        create_threshold_plot(
            data_dict=data, 
            estimated_threshold=avg_threshold, 
            output_filename=final_path
        )
        print(f" Plot saved locally as: {final_path}")

    except Exception as e:
        print("ERROR:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
