/** A small demo: two customer lists with typos, missing fields and one non-match. */
export const DEMO_LEFT = [
  { name: 'Jonathan Smith', city: 'Paris', email: 'jsmith@example.com' },
  { name: 'Katherine Doe', city: 'Lyon', email: 'kdoe@example.org' },
  { name: 'Ann Lee', city: 'Nice', email: '' },
  { name: 'Bob Martin', city: 'Lille', email: 'bob@martin.fr' },
  { name: 'Robert Martin', city: 'Lyon', email: 'rmartin@example.com' },
  { name: 'Chloé Dubois', city: 'Bordeaux', email: 'chloe.d@example.fr' },
];

export const DEMO_RIGHT = [
  { name: 'Kathryn Doe', city: 'Lyon', email: 'kdoe@example.org' },
  { name: 'Jon Smith', city: 'Paris', email: 'j.smith@example.com' },
  { name: 'Anne Li', city: 'Nice', email: 'anne.li@example.com' },
  { name: 'Rob Martin', city: 'Lyon', email: 'rmartin@example.com' },
  { name: 'B. Martin', city: 'Lille', email: 'bob@martin.fr' },
  { name: 'Chloe Dubois', city: 'Bordeaux', email: 'chloe.d@example.fr' },
  { name: 'Zoe Bernard', city: 'Toulouse', email: 'zoe@example.net' },
];

export const toJsonLines = (items: object[]) => items.map((x) => JSON.stringify(x)).join('\n');
