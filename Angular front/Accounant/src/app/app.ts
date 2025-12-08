import { Component, signal } from '@angular/core';
import { FinancialView } from './components/financial-view/financial-view';

@Component({
  selector: 'app-root',
  imports: [FinancialView],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('Accounant');
}
