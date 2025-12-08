export class Statistics
{
    //general statistics-----------------------------
    public comparison_days: number;
    public missing_days: number;
    public overall_accuracy: number;
    public correct_predictions: number;
    public incorrect_predictions: number;
    //general statistics-----------------------------

    //error statistics-------------------------------
    public mean_absolute_error: number;
    public mean_squared_error: number;
    public root_mean_squared_error: number;
    public mean_absolute_percentage_error: number;
    public error_standard_deviation: number;
    public r_squared: number;
    public directional_accuracy: number;
    //error statistics-------------------------------

    constructor(comparison_days: number,  missing_days: number, overall_accuracy: number, 
        correct_predictions: number, incorrect_predictions: number, mean_absolute_error: number,mean_squared_error: number, 
        root_mean_squared_error: number, mean_absolute_percentage_error: number, error_standard_deviation: number, r_squared: number, 
        directional_accuracy: number)
    {
        this.comparison_days = comparison_days;
        this.missing_days = missing_days;
        this.overall_accuracy = overall_accuracy;
        this.correct_predictions = correct_predictions;
        this.incorrect_predictions = incorrect_predictions;

        this.mean_absolute_error = mean_absolute_error;
        this.mean_squared_error = mean_squared_error;
        this.root_mean_squared_error = root_mean_squared_error;
        this.mean_absolute_percentage_error = mean_absolute_percentage_error;
        this.error_standard_deviation = error_standard_deviation;
        this.r_squared = r_squared;
        this.directional_accuracy = directional_accuracy;
    }
}

  