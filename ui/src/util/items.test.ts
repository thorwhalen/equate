import { describe, expect, it } from 'vitest';
import { displayText, parseItems } from './items';

describe('parseItems', () => {
  it('reads plain lines, JSON arrays and JSON lines', () => {
    expect(parseItems('apple\n\n banana \n')).toEqual(['apple', 'banana']);
    expect(parseItems('[{"a":1},"x"]')).toEqual([{ a: 1 }, 'x']);
    expect(parseItems('{"n":"a"}\n{"n":"b"}')).toEqual([{ n: 'a' }, { n: 'b' }]);
    expect(parseItems('  ')).toEqual([]);
  });
  it('rejects a JSON value that is not an array', () => {
    expect(() => parseItems('[1')).toThrow();
  });
});

describe('displayText', () => {
  it('joins record values', () => {
    expect(displayText({ name: 'Jon', city: null, zip: 75 })).toBe('Jon · 75');
    expect(displayText('x')).toBe('x');
  });
});
