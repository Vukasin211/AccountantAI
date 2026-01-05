import { Component } from '@angular/core';
import { FormControl, Validators, FormsModule, ReactiveFormsModule, FormGroup } from '@angular/forms';
import { HttpClient, HttpParams } from '@angular/common/http';
import { finalize, map, Observable, of, switchMap, shareReplay, delay } from 'rxjs';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatRadioModule } from '@angular/material/radio';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-compare-view',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatRadioModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatCheckboxModule
  ],
  templateUrl: './compare-view.html',
  styleUrls: ['./compare-view.css'],
})
export class CompareView {

  simulationMode: 'tomorrow' | 'tomorrowAuto' | 'simulate' = 'tomorrow';

  tomorrow = { currentDate: '' };
  simulate = { max_days: 30, overestimate_ok: false };

  accuracyCtrl = new FormControl<number>(30, {
    nonNullable: true,
    validators: [Validators.required, Validators.min(0), Validators.max(100)],
  });

  form = new FormGroup({ accuracy: this.accuracyCtrl });

  // Graphs
  imageUrl1$: Observable<string | null> | null = null;
  imageUrl2$: Observable<string | null> | null = null;
  imageLoading1 = false;
  imageLoading2 = false;

  // Data (JSON) for both
  simulationData1$: Observable<any> | null = null;
  simulationData2$: Observable<any> | null = null;

  constructor(private http: HttpClient) {}

  onSubmit() {
    this.imageUrl1$ = null;
    this.imageUrl2$ = null;

    const accuracyParam = (this.accuracyCtrl.value! / 100).toString();

    let url1 = '';
    let url2 = '';
    let baseParams1: any = {};
    let baseParams2: any = {};

    // Set base parameters & urls based on simulation mode
    if (this.simulationMode === 'simulate') {
      baseParams1 = {
        max_days: this.simulate.max_days.toString(),
        accuracy: accuracyParam,
        overestimate_ok: this.simulate.overestimate_ok ? '1' : '0'
      };
      baseParams2 = { ...baseParams1 }; // identical for second URL

      url1 = `${environment.apiUrl}simulate`;
      url2 = `${environment.apiUrl}simulateSingle`;

    } else if (this.simulationMode === 'tomorrow') {
      baseParams1 = { currentDateTime: this.tomorrow.currentDate, accuracy: accuracyParam };
      baseParams2 = { currentDateTime: this.tomorrow.currentDate, accuracy: accuracyParam };

      url1 = `${environment.apiUrl}simulateTomorrow`;
      url2 = `${environment.apiUrl}simulateTomorrowSingle`;

    } else if (this.simulationMode === 'tomorrowAuto') {
      baseParams1 = { currentDateTime: this.tomorrow.currentDate, accuracy: accuracyParam };
      baseParams2 = { currentDateTime: this.tomorrow.currentDate, accuracy: accuracyParam };

      url1 = `${environment.apiUrl}simulateTomorrowAuto`;
      url2 = `${environment.apiUrl}simulateTomorrowAutoSingle`;
    }

    // IMAGE params (return_graph=1)
    const httpParamsImage1 = new HttpParams({ fromObject: { ...baseParams1, return_graph: '1' } });
    const httpParamsImage2 = new HttpParams({ fromObject: { ...baseParams2, return_graph: '1' } });

    // DATA params (return_graph=0)
    const httpParamsData1 = new HttpParams({ fromObject: { ...baseParams1, return_graph: '0' } });
    const httpParamsData2 = new HttpParams({ fromObject: { ...baseParams2, return_graph: '0' } });

    // --- IMAGE REQUESTS ---
    this.imageLoading1 = true;
    this.imageLoading2 = true;

    this.imageUrl1$ = this.http.post(url1, null, { params: httpParamsImage1, responseType: 'blob' }).pipe(
      map(blob => blob.type.startsWith('image/') ? URL.createObjectURL(blob) : null),
      finalize(() => this.imageLoading1 = false),
      switchMap(image1Url => {
        // Request second image after first completes
        this.imageUrl2$ = this.http.post(url2, null, { params: httpParamsImage2, responseType: 'blob' }).pipe(
          delay(50),
          map(blob => blob.type.startsWith('image/') ? URL.createObjectURL(blob) : null),
          finalize(() => this.imageLoading2 = false),
          shareReplay(1)
        );
        return of(image1Url);
      }),
      shareReplay(1)
    );

    // --- DATA REQUESTS (parallel) ---
    this.simulationData1$ = this.http.get(url1, { params: httpParamsData1 }).pipe(shareReplay(1));
    this.simulationData2$ = this.http.get(url2, { params: httpParamsData2 }).pipe(shareReplay(1));
  }
}