import { describe, expect, it } from 'vitest';
import { reviewColumns } from './collection';

describe('review collection', () => {
  it('generates one column per row field', () => {
    expect(reviewColumns.map((c) => c.id)).toEqual(['item', 'left', 'match', 'score', 'margin', 'status']);
  });
});
