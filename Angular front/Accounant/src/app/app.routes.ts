import { Routes } from '@angular/router';
import { FinancialView } from './components/financial-view/financial-view';
import { CompareView } from './components/compare-view/compare-view';

export const routes: Routes = [
  { path: '', component: FinancialView },
  { path: 'compare', component: CompareView }
];