import { Component } from '@angular/core';
import { FormControl, Validators, FormsModule, ReactiveFormsModule, FormGroup } from '@angular/forms';

import { MatCardModule } from '@angular/material/card';
import { MatRadioModule } from '@angular/material/radio';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSliderModule } from '@angular/material/slider';
import { MatButtonModule } from '@angular/material/button';
import { MatExpansionModule } from '@angular/material/expansion';
import { CommonModule } from '@angular/common';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { environment } from '../../../environments/environment';
import { HttpClient } from '@angular/common/http';
import { finalize, map, Observable, shareReplay, switchMap } from 'rxjs';
import { Router } from '@angular/router';

@Component({
  selector: 'app-financial-view',
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatRadioModule,
    MatFormFieldModule,
    MatInputModule,
    MatSliderModule,
    MatButtonModule,
    MatExpansionModule,
    MatCheckboxModule
  ],
  templateUrl: './financial-view.html',
  styleUrl: './financial-view.css',
})
export class FinancialView {
  constructor(private http: HttpClient, private router: Router) {}

  selectedModel: 'optimal' | 'single' = 'single';

  accuracyCtrl = new FormControl<number>(30, {
    nonNullable: true,
    validators: [Validators.required, Validators.min(0), Validators.max(100)],
  });

  simulationMode: 'tomorrow' | 'tomorrowAuto' | 'simulate' = 'tomorrow';

  tomorrow = { currentDate: '' };
  tomorrowAuto = {};
  simulate = { max_days: 30, overestimate_ok: 0 };

  form = new FormGroup({ accuracy: this.accuracyCtrl });

  imageUrl$: Observable<string | null> | null = null;
  simulationDays$: Observable<any> | null = null;
  simulationErrors$ : Observable<any> | null = null;
  isSubmitting = false;
  isLoadingDays = false;

  onSubmit() {
    if (this.selectedModel === 'optimal' && this.simulationMode === 'simulate') {
      const params = {
        max_days: this.simulate.max_days.toString(),
        accuracy: (this.accuracyCtrl.value! / 100).toString(),
        overestimate_ok: this.simulate.overestimate_ok ? '1' : '0',
      };

      this.isSubmitting = true;
      this.isLoadingDays = true;

      // IMAGE request
      this.imageUrl$ = this.http
        .post(`${environment.apiUrl}simulate`, null, { params, responseType: 'blob' })
        .pipe(
          map((blob: Blob) =>
            blob.type.startsWith('image/') ? URL.createObjectURL(blob) : null
          ),
          finalize(() => (this.isSubmitting = false)),
          shareReplay(1)
        );

      // DAYS request (starts immediately)
      this.simulationDays$ = this.http
        .get(`${environment.apiUrl}simulate-json`, { params })
        .pipe(finalize(() => (this.isLoadingDays = false)));

      this.simulationErrors$ = this.simulationDays$.pipe(
        map((res: any) => res?.error_statistics ?? null)
      );
    }
    else if (this.selectedModel === 'single' && this.simulationMode === 'simulate') {
      const params = {
        max_days: this.simulate.max_days.toString(),
        accuracy: (this.accuracyCtrl.value! / 100).toString(),
        overestimate_ok: this.simulate.overestimate_ok ? '1' : '0',
      };

      this.isSubmitting = true;
      this.isLoadingDays = true;

      // IMAGE request
      this.imageUrl$ = this.http
        .post(`${environment.apiUrl}simulateSingle`, null, { params, responseType: 'blob' })
        .pipe(
          map((blob: Blob) =>
            blob.type.startsWith('image/') ? URL.createObjectURL(blob) : null
          ),
          finalize(() => (this.isSubmitting = false)),
          shareReplay(1)
        );

      // DAYS request (starts immediately)
      this.simulationDays$ = this.http
        .get(`${environment.apiUrl}simulate-json-single`, { params })
        .pipe(finalize(() => (this.isLoadingDays = false)));

       this.simulationErrors$ = this.simulationDays$.pipe(
          map((res: any) => res?.error_statistics ?? null)
        ); 
    }
    else if (this.selectedModel === 'optimal' && this.simulationMode === 'tomorrowAuto') {
            const baseParams = {
        accuracy: (this.accuracyCtrl.value! / 100).toString(),
      };

      this.isSubmitting = true;
      this.isLoadingDays = true;

      // IMAGE request
      this.imageUrl$ = this.http
        .post(
          `${environment.apiUrl}simulateTomorrowAuto`,
          null,
          {
            params: { ...baseParams, return_graph: '1' },
            responseType: 'blob',
          }
        )
        .pipe(
          map((blob: Blob) =>
            blob.type.startsWith('image/')
              ? URL.createObjectURL(blob)
              : null
          ),
          finalize(() => (this.isSubmitting = false)),
          shareReplay(1)
        );

      // DAYS request
      this.simulationDays$ = this.http
        .post(
          `${environment.apiUrl}simulateTomorrowAuto`,
          null,
          {
            params: { ...baseParams, return_graph: '0' },
          }
        )
        .pipe(finalize(() => (this.isLoadingDays = false)));

        this.simulationErrors$ = this.simulationDays$.pipe(
          map((res: any) => res?.error_statistics ?? null)
        );
    }
    else if (this.selectedModel === 'single' && this.simulationMode === 'tomorrowAuto') {
        const baseParams = {
        accuracy: (this.accuracyCtrl.value! / 100).toString(),
      };

      this.isSubmitting = true;
      this.isLoadingDays = true;

      // IMAGE request
      this.imageUrl$ = this.http
        .post(
          `${environment.apiUrl}simulateTomorrowAutoSingle`,
          null,
          {
            params: { ...baseParams, return_graph: '1' },
            responseType: 'blob',
          }
        )
        .pipe(
          map((blob: Blob) =>
            blob.type.startsWith('image/')
              ? URL.createObjectURL(blob)
              : null
          ),
          finalize(() => (this.isSubmitting = false)),
          shareReplay(1)
        );

      // DAYS request
      this.simulationDays$ = this.http
        .post(
          `${environment.apiUrl}simulateTomorrowAutoSingle`,
          null,
          {
            params: { ...baseParams, return_graph: '0' },
          }
        )
        .pipe(finalize(() => (this.isLoadingDays = false)));

        this.simulationErrors$ = this.simulationDays$.pipe(
          map((res: any) => res?.error_statistics ?? null)
        );
    }
    else if (this.selectedModel === 'optimal' && this.simulationMode === 'tomorrow') {
      const baseParams = {
        currentDateTime: this.tomorrow.currentDate, // ✅ USE USER INPUT
        accuracy: (this.accuracyCtrl.value! / 100).toString(),
      };

      this.isSubmitting = true;
      this.isLoadingDays = true;

      // IMAGE request
      this.imageUrl$ = this.http
        .post(
          `${environment.apiUrl}simulateTomorrow`,
          null,
          {
            params: { ...baseParams, return_graph: '1' },
            responseType: 'blob',
          }
        )
        .pipe(
          map((blob: Blob) =>
            blob.type.startsWith('image/')
              ? URL.createObjectURL(blob)
              : null
          ),
          finalize(() => (this.isSubmitting = false)),
          shareReplay(1)
        );

      // DAYS request
      this.simulationDays$ = this.http
        .post(
          `${environment.apiUrl}simulateTomorrow`,
          null,
          {
            params: { ...baseParams, return_graph: '0' },
          }
        )
        .pipe(finalize(() => (this.isLoadingDays = false)));

        this.simulationErrors$ = this.simulationDays$.pipe(
          map((res: any) => res?.error_statistics ?? null)
        );
    }
    else if (this.selectedModel === 'single' && this.simulationMode === 'tomorrow') {
              const baseParams = {
          currentDateTime: this.tomorrow.currentDate, 
          accuracy: (this.accuracyCtrl.value! / 100).toString(),
        };

        this.isSubmitting = true;
        this.isLoadingDays = true;

        // IMAGE request
        this.imageUrl$ = this.http
          .post(
            `${environment.apiUrl}simulateTomorrowSingle`,
            null,
            {
              params: { ...baseParams, return_graph: '1' },
              responseType: 'blob',
            }
          )
          .pipe(
            map((blob: Blob) =>
              blob.type.startsWith('image/')
                ? URL.createObjectURL(blob)
                : null
            ),
            finalize(() => (this.isSubmitting = false)),
            shareReplay(1)
          );

        // DAYS request
        this.simulationDays$ = this.http
          .post(
            `${environment.apiUrl}simulateTomorrowSingle`,
            null,
            {
              params: { ...baseParams, return_graph: '0' },
            }
          )
          .pipe(finalize(() => (this.isLoadingDays = false)));

          this.simulationErrors$ = this.simulationDays$.pipe(
            map((res: any) => res?.error_statistics ?? null)
          );
    }
  }

  onReset() {
    this.form.reset({ accuracy: 30 });
    this.selectedModel = 'single';
    this.tomorrow.currentDate = '';
    this.simulate.max_days = 30;
    this.simulate.overestimate_ok = 0;
    this.imageUrl$ = null;
    this.isSubmitting = false;
    this.simulationDays$ = null;
    this.isLoadingDays = false;
    this.simulationErrors$ = null;
  }
  
  onCompare() {
    this.router.navigate(['/compare']);
  }
}