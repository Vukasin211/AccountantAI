export class Day
{
    public day: number;
    public dateValue: string;
    public predicted_amount: number;
    public actual_amount: number | null;
    public model_used: string;
    public correct: boolean;
    public window_tail: number[];

    constructor(day: number, dateValue: string, predicted_amount: number, actual_amount: number | null, model_used: string, correct: boolean, 
        window_tail: number[])
    {
        this.day = day;
        this.dateValue = dateValue;
        this.predicted_amount = predicted_amount;
        this.actual_amount = actual_amount
        this.model_used = model_used;
        this.correct = correct;
        this.window_tail = window_tail;
    }
}


