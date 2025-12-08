import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FinancialView } from './financial-view';

describe('FinancialView', () => {
  let component: FinancialView;
  let fixture: ComponentFixture<FinancialView>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FinancialView]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FinancialView);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
