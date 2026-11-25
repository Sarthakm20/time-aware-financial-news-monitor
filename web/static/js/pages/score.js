// Placeholder until this page is built (plan Tasks 7-11).
import { card, emptyState } from "../ui.js";

export async function render(root) {
  root.replaceChildren(card({ body: emptyState("This page is being built.") }));
}
