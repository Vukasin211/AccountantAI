import { Component, signal } from '@angular/core';
import { FinancialView } from './components/financial-view/financial-view';
import { RouterOutlet } from "@angular/router";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet], // NO AppRoutingModule here!
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {
  protected readonly title = signal('Accounant');
}