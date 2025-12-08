export class Metadata
{
    public day: string;
    public first_prediction_date: string;
    public max_days: number;
    public accuracy_threshold: number;
    public overestimate_ok: boolean;
    public simulation_period: string;

    constructor(day: string, first_prediction_date: string, max_days: number, accuracy_threshold: number, 
        overestimate_ok: boolean, 
        simulation_period: string)
    {
        this.day = day;
        this.first_prediction_date = first_prediction_date;
        this.max_days = max_days;
        this.accuracy_threshold = accuracy_threshold;
        this.overestimate_ok = overestimate_ok;
        this.simulation_period = simulation_period;
    }
}
