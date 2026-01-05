import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CompareView } from './compare-view';

describe('CompareView', () => {
  let component: CompareView;
  let fixture: ComponentFixture<CompareView>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CompareView]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CompareView);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
